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
import unicodedata
from collections import Counter
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from typing import TypeVar

import zstandard as zstd

DEFAULT_REPETITION_NGRAM_SIZE = 16
DEFAULT_REPETITION_MAX_COUNT = 5
DEFAULT_CONSECUTIVE_MIN_PATTERN_SIZE = 4
DEFAULT_CONSECUTIVE_MAX_PATTERN_SIZE = 64
DEFAULT_CONSECUTIVE_MIN_COUNT = 3
DEFAULT_REPETITION_FALLBACK_MAX_TOKENS: int | None = None
DEFAULT_REPETITION_TEXT_MIN_BYTES = 256
DEFAULT_REPETITION_TEXT_CHECK_EVERY_TOKENS = 32
DEFAULT_REPETITION_FULL_TEXT_MAX_BYTES = 8192
DEFAULT_REPETITION_ZSTD_LEVEL = 3
DEFAULT_REPETITION_ZSTD_MIN_RATIO = 0.18
DEFAULT_REPETITION_ZSTD_MAX_RATIO: float | None = None
DEFAULT_REPETITION_ZSTD_WINDOW_BYTES = 2048
DEFAULT_REPETITION_ZSTD_WINDOW_MIN_TOTAL_BYTES = 8192
DEFAULT_REPETITION_ZSTD_WINDOW_MIN_RATIO = 0.40
DEFAULT_REPETITION_SCRIPT_MIN_CHARS = 256
DEFAULT_REPETITION_SCRIPT_MIN_SUSPICIOUS_COUNT = 3
DEFAULT_REPETITION_SCRIPT_MIN_KINDS = 3
DEFAULT_REPETITION_SCRIPT_MIN_RATIO = 0.005
DEFAULT_REPETITION_SCRIPT_DENSE_MIN_COUNT = 15
DEFAULT_REPETITION_SCRIPT_DENSE_MIN_RATIO = 0.01
DEFAULT_REPETITION_SCRIPT_WINDOW_CHARS = 2048
DEFAULT_REPETITION_TEXT_NGRAM_SIZE = 6
DEFAULT_REPETITION_TEXT_NGRAM_MIN_COUNT = 10
DEFAULT_REPETITION_TEXT_NGRAM_MIN_BYTES = 8192
DEFAULT_REPETITION_TEXT_NGRAM_WINDOW_CHARS = 4096
DEFAULT_REPETITION_REASONING_MARKER_WINDOW_CHARS = 4096
DEFAULT_REPETITION_REASONING_MARKER_MIN_COUNT = 25
DEFAULT_REPETITION_REASONING_MARKERS = (
    "wait",
    "maybe",
    "let's",
    "alternatively",
    "not sure",
    "confused",
    "restart",
    "again carefully",
    "re-start",
    "start over",
    "another approach",
)
DEFAULT_REPETITION_RULES: tuple[tuple[int, int], ...] = (
    (16, 6),
    (12, 5),
    (8, 10),
    (6, 12),
    (5, 14),
    (4, 20),
)
REPETITION_EXTRA_FIELD_KEYS: tuple[str, ...] = (
    "repetition_truncated",
    "repetition_ngram_size",
    "repetition_ngram_count_threshold",
    "repetition_fallback_max_tokens",
    "repetition_detection_rules",
    "repetition_matched_rule",
    "repetition_matched_reason",
    "repetition_truncation_length",
    "original_response_length",
    "repetition_text_byte_length",
    "repetition_zstd_ratio",
    "repetition_zstd_window_ratio",
    "repetition_script_counts",
    "repetition_suspicious_script_counts",
    "repetition_suspicious_script_ratio",
    "repetition_script_window_suspicious_counts",
    "repetition_script_window_suspicious_ratio",
    "repetition_rare_fragmented_scripts",
    "repetition_dense_suspicious_scripts",
    "repetition_script_window_dense_suspicious_scripts",
    "repetition_replacement_count",
    "repetition_text_ngram_size",
    "repetition_text_ngram_max_count",
    "repetition_text_ngram_match",
    "repetition_reasoning_marker_count",
    "repetition_reasoning_marker_counts",
    "repetition_seen_think_close",
    "repetition_starts_in_think",
    "repetition_unclosed_think_detected",
)

T = TypeVar("T")


_SCRIPT_MIX_ALLOWED = frozenset({"Latin", "CJK", "Greek"})
_TEXT_NGRAM_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[\u4e00-\u9fff]|[^\s\w]", re.UNICODE)


def _char_script(ch: str) -> str | None:
    if ch == "\ufffd":
        return "Replacement"

    codepoint = ord(ch)
    category = unicodedata.category(ch)
    if category[0] in {"Z", "P", "S", "N", "C"}:
        return None

    if 0x4E00 <= codepoint <= 0x9FFF or 0x3400 <= codepoint <= 0x4DBF or 0x20000 <= codepoint <= 0x2A6DF:
        return "CJK"
    if 0x3040 <= codepoint <= 0x309F:
        return "Hiragana"
    if 0x30A0 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF:
        return "Katakana"
    if 0xAC00 <= codepoint <= 0xD7AF or 0x1100 <= codepoint <= 0x11FF:
        return "Hangul"
    if 0x0400 <= codepoint <= 0x052F:
        return "Cyrillic"
    if 0x0370 <= codepoint <= 0x03FF:
        return "Greek"
    if 0x0600 <= codepoint <= 0x06FF or 0x0750 <= codepoint <= 0x077F:
        return "Arabic"
    if 0x0900 <= codepoint <= 0x097F:
        return "Devanagari"
    if 0x0590 <= codepoint <= 0x05FF:
        return "Hebrew"
    if 0x0E00 <= codepoint <= 0x0E7F:
        return "Thai"

    name = unicodedata.name(ch, "")
    if 0x1D400 <= codepoint <= 0x1D7FF or 0x2100 <= codepoint <= 0x214F:
        if "GREEK" in name:
            return "Greek"
        if ch.isalpha():
            return "Latin"
    if name.startswith("LATIN") or ("LATIN" in name and ch.isalpha()):
        return "Latin"
    if ch.isalpha():
        return name.split()[0] if name else "OtherAlpha"
    return None


def _script_runs(scripts: Sequence[str]) -> dict[str, list[int]]:
    runs: dict[str, list[int]] = {}
    index = 0
    while index < len(scripts):
        script = scripts[index]
        next_index = index + 1
        while next_index < len(scripts) and scripts[next_index] == script:
            next_index += 1
        runs.setdefault(script, []).append(next_index - index)
        index = next_index
    return runs


def _normalize_rules(rules: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    for ngram_size, min_count in rules:
        if ngram_size <= 0:
            raise ValueError(f"ngram_size must be positive, got {ngram_size}")
        if min_count < 2:
            raise ValueError(f"min_count must be at least 2, got {min_count}")
        normalized.append((int(ngram_size), int(min_count)))
    if not normalized:
        raise ValueError("at least one repetition detection rule is required")
    return tuple(normalized)


class ConsecutiveRepetitionDetector:
    """Detect exact token blocks repeated consecutively at the generated tail."""

    def __init__(
        self,
        *,
        min_pattern_size: int = DEFAULT_CONSECUTIVE_MIN_PATTERN_SIZE,
        max_pattern_size: int = DEFAULT_CONSECUTIVE_MAX_PATTERN_SIZE,
        min_count: int = DEFAULT_CONSECUTIVE_MIN_COUNT,
    ) -> None:
        if min_pattern_size <= 0:
            raise ValueError(f"min_pattern_size must be positive, got {min_pattern_size}")
        if max_pattern_size < min_pattern_size:
            raise ValueError(
                f"max_pattern_size must be >= min_pattern_size, got {max_pattern_size} < {min_pattern_size}"
            )
        if min_count < 2:
            raise ValueError(f"min_count must be at least 2, got {min_count}")
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.min_count = min_count
        self._next_length = min_pattern_size * min_count
        self.truncation_length: int | None = None
        self.matched_rule: tuple[int, int] | None = None
        self.matched_reason: str | None = None
        self.matched_text_stats: dict[str, object] = {}
        self.last_text_stats: dict[str, object] = {}

    def _tail_repeats(self, token_ids: Sequence[int], *, end: int, pattern_size: int) -> bool:
        for offset in range(1, pattern_size + 1):
            target_token = token_ids[end - offset]
            for repeat_index in range(1, self.min_count):
                if token_ids[end - repeat_index * pattern_size - offset] != target_token:
                    return False
        return True

    def observe(self, token_ids: Sequence[int]) -> int | None:
        if self.truncation_length is not None:
            return self.truncation_length

        while self._next_length <= len(token_ids):
            end = self._next_length
            max_pattern_size = min(self.max_pattern_size, end // self.min_count)
            for pattern_size in range(self.min_pattern_size, max_pattern_size + 1):
                if self._tail_repeats(token_ids, end=end, pattern_size=pattern_size):
                    self.truncation_length = end
                    self.matched_rule = (pattern_size, self.min_count)
                    self.matched_reason = "consecutive_ngram"
                    return end
            self._next_length += 1

        return None


class NGramRepetitionDetector:
    def __init__(
        self,
        ngram_size: int = DEFAULT_REPETITION_NGRAM_SIZE,
        max_count: int = DEFAULT_REPETITION_MAX_COUNT,
        rules: Sequence[tuple[int, int]] | None = None,
        fallback_max_tokens: int | None = DEFAULT_REPETITION_FALLBACK_MAX_TOKENS,
        token_to_bytes: Callable[[int], bytes] | None = None,
        text_min_bytes: int = DEFAULT_REPETITION_TEXT_MIN_BYTES,
        text_check_every_tokens: int = DEFAULT_REPETITION_TEXT_CHECK_EVERY_TOKENS,
        full_text_max_bytes: int = DEFAULT_REPETITION_FULL_TEXT_MAX_BYTES,
        zstd_level: int = DEFAULT_REPETITION_ZSTD_LEVEL,
        zstd_min_ratio: float = DEFAULT_REPETITION_ZSTD_MIN_RATIO,
        zstd_max_ratio: float | None = DEFAULT_REPETITION_ZSTD_MAX_RATIO,
        zstd_window_bytes: int = DEFAULT_REPETITION_ZSTD_WINDOW_BYTES,
        zstd_window_min_total_bytes: int = DEFAULT_REPETITION_ZSTD_WINDOW_MIN_TOTAL_BYTES,
        zstd_window_min_ratio: float = DEFAULT_REPETITION_ZSTD_WINDOW_MIN_RATIO,
        script_min_chars: int = DEFAULT_REPETITION_SCRIPT_MIN_CHARS,
        script_min_suspicious_count: int = DEFAULT_REPETITION_SCRIPT_MIN_SUSPICIOUS_COUNT,
        script_min_kinds: int = DEFAULT_REPETITION_SCRIPT_MIN_KINDS,
        script_min_ratio: float = DEFAULT_REPETITION_SCRIPT_MIN_RATIO,
        script_dense_min_count: int = DEFAULT_REPETITION_SCRIPT_DENSE_MIN_COUNT,
        script_dense_min_ratio: float = DEFAULT_REPETITION_SCRIPT_DENSE_MIN_RATIO,
        script_window_chars: int = DEFAULT_REPETITION_SCRIPT_WINDOW_CHARS,
        text_ngram_size: int = DEFAULT_REPETITION_TEXT_NGRAM_SIZE,
        text_ngram_min_count: int = DEFAULT_REPETITION_TEXT_NGRAM_MIN_COUNT,
        text_ngram_min_bytes: int = DEFAULT_REPETITION_TEXT_NGRAM_MIN_BYTES,
        text_ngram_window_chars: int = DEFAULT_REPETITION_TEXT_NGRAM_WINDOW_CHARS,
        reasoning_marker_window_chars: int = DEFAULT_REPETITION_REASONING_MARKER_WINDOW_CHARS,
        reasoning_marker_min_count: int = DEFAULT_REPETITION_REASONING_MARKER_MIN_COUNT,
        reasoning_markers: Sequence[str] = DEFAULT_REPETITION_REASONING_MARKERS,
    ) -> None:
        if ngram_size <= 0:
            raise ValueError(f"ngram_size must be positive, got {ngram_size}")
        if max_count <= 0:
            raise ValueError(f"max_count must be positive, got {max_count}")
        if fallback_max_tokens is not None and fallback_max_tokens <= 0:
            raise ValueError(f"fallback_max_tokens must be positive, got {fallback_max_tokens}")
        if text_min_bytes <= 0:
            raise ValueError(f"text_min_bytes must be positive, got {text_min_bytes}")
        if text_check_every_tokens <= 0:
            raise ValueError(f"text_check_every_tokens must be positive, got {text_check_every_tokens}")
        if full_text_max_bytes <= 0:
            raise ValueError(f"full_text_max_bytes must be positive, got {full_text_max_bytes}")
        if zstd_min_ratio < 0:
            raise ValueError(f"zstd_min_ratio must be non-negative, got {zstd_min_ratio}")
        if zstd_max_ratio is not None and zstd_max_ratio <= zstd_min_ratio:
            raise ValueError(
                f"zstd_max_ratio must be greater than zstd_min_ratio, got {zstd_max_ratio} <= {zstd_min_ratio}"
            )
        if zstd_window_bytes <= 0:
            raise ValueError(f"zstd_window_bytes must be positive, got {zstd_window_bytes}")
        if zstd_window_min_total_bytes <= 0:
            raise ValueError(f"zstd_window_min_total_bytes must be positive, got {zstd_window_min_total_bytes}")
        if zstd_window_min_ratio < 0:
            raise ValueError(f"zstd_window_min_ratio must be non-negative, got {zstd_window_min_ratio}")
        if script_window_chars <= 0:
            raise ValueError(f"script_window_chars must be positive, got {script_window_chars}")
        if script_dense_min_count <= 0:
            raise ValueError(f"script_dense_min_count must be positive, got {script_dense_min_count}")
        if script_dense_min_ratio < 0:
            raise ValueError(f"script_dense_min_ratio must be non-negative, got {script_dense_min_ratio}")
        if text_ngram_size <= 0:
            raise ValueError(f"text_ngram_size must be positive, got {text_ngram_size}")
        if text_ngram_min_count < 2:
            raise ValueError(f"text_ngram_min_count must be at least 2, got {text_ngram_min_count}")
        if text_ngram_min_bytes <= 0:
            raise ValueError(f"text_ngram_min_bytes must be positive, got {text_ngram_min_bytes}")
        if text_ngram_window_chars <= 0:
            raise ValueError(f"text_ngram_window_chars must be positive, got {text_ngram_window_chars}")
        if reasoning_marker_window_chars <= 0:
            raise ValueError(f"reasoning_marker_window_chars must be positive, got {reasoning_marker_window_chars}")
        if reasoning_marker_min_count <= 0:
            raise ValueError(f"reasoning_marker_min_count must be positive, got {reasoning_marker_min_count}")
        self.ngram_size = ngram_size
        self.max_count = max_count
        self.fallback_max_tokens = fallback_max_tokens
        self.token_to_bytes = token_to_bytes
        self.text_min_bytes = text_min_bytes
        self.text_check_every_tokens = text_check_every_tokens
        self.full_text_max_bytes = full_text_max_bytes
        self.zstd_min_ratio = zstd_min_ratio
        self.zstd_max_ratio = zstd_max_ratio
        self.zstd_window_bytes = zstd_window_bytes
        self.zstd_window_min_total_bytes = zstd_window_min_total_bytes
        self.zstd_window_min_ratio = zstd_window_min_ratio
        self.script_min_chars = script_min_chars
        self.script_min_suspicious_count = script_min_suspicious_count
        self.script_min_kinds = script_min_kinds
        self.script_min_ratio = script_min_ratio
        self.script_dense_min_count = script_dense_min_count
        self.script_dense_min_ratio = script_dense_min_ratio
        self.script_window_chars = script_window_chars
        self.text_ngram_size = text_ngram_size
        self.text_ngram_min_count = text_ngram_min_count
        self.text_ngram_min_bytes = text_ngram_min_bytes
        self.text_ngram_window_chars = text_ngram_window_chars
        self.reasoning_marker_window_chars = reasoning_marker_window_chars
        self.reasoning_marker_min_count = reasoning_marker_min_count
        self.reasoning_markers = tuple(marker.lower() for marker in reasoning_markers)
        self._zstd_compressor = zstd.ZstdCompressor(level=zstd_level)
        if rules is None and ngram_size == DEFAULT_REPETITION_NGRAM_SIZE and max_count == DEFAULT_REPETITION_MAX_COUNT:
            rules = DEFAULT_REPETITION_RULES
        elif rules is None:
            rules = ((ngram_size, max_count + 1),)
        self.rules = _normalize_rules(rules)
        self._counts: dict[int, dict[tuple[int, ...], int]] = {}
        self._next_start: dict[int, int] = {}
        self._text_next_token_index = 0
        self._text_last_check_token_index = 0
        self._text_bytes = bytearray()
        self._seen_think_close = False
        self.truncation_length: int | None = None
        self.matched_rule: tuple[int, int] | None = None
        self.matched_reason: str | None = None
        self.matched_text_stats: dict[str, object] = {}
        self.last_text_stats: dict[str, object] = {}

    def observe(self, token_ids: Sequence[int]) -> int | None:
        if self.truncation_length is not None:
            return self.truncation_length

        candidates: list[tuple[int, tuple[int, int] | None, str, dict[str, object]]] = []
        for ngram_size, min_count in self.rules:
            counts = self._counts.setdefault(ngram_size, {})
            next_start = self._next_start.get(ngram_size, 0)
            last_start = len(token_ids) - ngram_size

            while next_start <= last_start:
                ngram = tuple(int(token_id) for token_id in token_ids[next_start : next_start + ngram_size])
                count = counts.get(ngram, 0) + 1
                counts[ngram] = count
                next_start += 1
                self._next_start[ngram_size] = next_start
                if count >= min_count:
                    candidates.append((next_start + ngram_size - 1, (ngram_size, min_count), "ngram_occurrence", {}))
                    break

        if self.fallback_max_tokens is not None and len(token_ids) >= self.fallback_max_tokens:
            candidates.append((self.fallback_max_tokens, None, "max_tokens_fallback", {}))

        text_candidate = self._observe_text_anomaly(token_ids)
        if text_candidate is not None:
            truncation_length, matched_reason, text_stats = text_candidate
            candidates.append((truncation_length, None, matched_reason, text_stats))

        if candidates:
            self.truncation_length, self.matched_rule, self.matched_reason, self.matched_text_stats = min(
                candidates, key=lambda item: item[0]
            )
            return self.truncation_length

        return None

    def _observe_text_anomaly(self, token_ids: Sequence[int]) -> tuple[int, str, dict[str, object]] | None:
        if self.token_to_bytes is None:
            return None

        while self._text_next_token_index < len(token_ids):
            token_id = int(token_ids[self._text_next_token_index])
            self._text_bytes.extend(self.token_to_bytes(token_id))
            self._text_next_token_index += 1
            candidate = self._check_text_anomaly()
            if candidate is not None:
                return candidate

        return None

    def _check_text_anomaly(self) -> tuple[int, str, dict[str, object]] | None:
        if len(self._text_bytes) < self.text_min_bytes:
            return None
        if self._text_next_token_index - self._text_last_check_token_index < self.text_check_every_tokens:
            return None

        self._text_last_check_token_index = self._text_next_token_index
        payload = bytes(self._text_bytes)
        window_payload = payload[-self.zstd_window_bytes :]
        window_compressed_size = len(self._zstd_compressor.compress(window_payload))
        zstd_window_ratio = window_compressed_size / len(window_payload) if window_payload else 1.0
        zstd_ratio: float | None = None
        script_stats = self._empty_script_stats()
        if len(payload) <= self.full_text_max_bytes:
            compressed_size = len(self._zstd_compressor.compress(payload))
            zstd_ratio = compressed_size / len(payload) if payload else 1.0
            script_stats = self._analyze_script_mix(payload.decode("utf-8", errors="replace"))

        text_window_bytes = max(
            self.zstd_window_bytes,
            self.script_window_chars * 4,
            self.text_ngram_window_chars * 4,
            self.reasoning_marker_window_chars * 4,
        )
        text_window = payload[-text_window_bytes:].decode("utf-8", errors="replace")
        if "</think>" in text_window:
            self._seen_think_close = True
        script_window_stats = self._analyze_script_mix(text_window[-self.script_window_chars :])
        text_ngram_stats = self._analyze_text_ngram_repetition(text_window[-self.text_ngram_window_chars :])
        reasoning_marker_stats = self._analyze_reasoning_marker_repetition(
            text_window[-self.reasoning_marker_window_chars :]
        )
        unclosed_think_stats = self._analyze_unclosed_think(payload)
        stats: dict[str, object] = {
            "text_byte_length": len(payload),
            "zstd_ratio": zstd_ratio,
            "zstd_window_byte_length": len(window_payload),
            "zstd_window_ratio": zstd_window_ratio,
            **script_stats,
            "script_window_counts": script_window_stats["script_counts"],
            "script_window_suspicious_counts": script_window_stats["suspicious_script_counts"],
            "script_window_suspicious_ratio": script_window_stats["suspicious_script_ratio"],
            "script_window_rare_fragmented_scripts": script_window_stats["rare_fragmented_scripts"],
            "script_window_dense_suspicious_scripts": script_window_stats["dense_suspicious_scripts"],
            "script_window_replacement_count": script_window_stats["replacement_count"],
            **text_ngram_stats,
            **reasoning_marker_stats,
            **unclosed_think_stats,
        }
        self.last_text_stats = stats

        if zstd_ratio is not None and zstd_ratio < self.zstd_min_ratio:
            return self._text_next_token_index, "zstd_low_ratio", stats
        if zstd_ratio is not None and self.zstd_max_ratio is not None and zstd_ratio > self.zstd_max_ratio:
            return self._text_next_token_index, "zstd_high_ratio", stats
        if len(payload) >= self.zstd_window_min_total_bytes and zstd_window_ratio < self.zstd_window_min_ratio:
            return self._text_next_token_index, "zstd_window_low_ratio", stats
        if script_stats["script_mix_detected"]:
            return self._text_next_token_index, "script_mix", stats
        if script_window_stats["script_mix_detected"]:
            return self._text_next_token_index, "script_window_mix", stats
        if (
            len(payload) >= self.text_ngram_min_bytes
            and text_ngram_stats["text_ngram_max_count"] >= self.text_ngram_min_count
        ):
            return self._text_next_token_index, "text_ngram_repetition", stats

        return None

    def _empty_script_stats(self) -> dict[str, object]:
        return {
            "script_counts": {},
            "suspicious_script_counts": {},
            "suspicious_script_ratio": 0.0,
            "rare_fragmented_scripts": [],
            "dense_suspicious_scripts": [],
            "replacement_count": 0,
            "script_mix_detected": False,
        }

    def _analyze_script_mix(self, text: str) -> dict[str, object]:
        scripts = [script for char in text if (script := _char_script(char)) is not None]
        script_counts: dict[str, int] = {}
        for script in scripts:
            script_counts[script] = script_counts.get(script, 0) + 1

        total = sum(script_counts.values())
        runs = _script_runs(scripts)
        rare_fragmented_scripts: list[dict[str, object]] = []
        for script, count in script_counts.items():
            if script in _SCRIPT_MIX_ALLOWED:
                continue
            script_runs = runs.get(script, [])
            if not script_runs or total == 0:
                continue
            ratio = count / total
            avg_run = count / len(script_runs)
            len1_run_ratio = sum(1 for run_len in script_runs if run_len == 1) / len(script_runs)
            adjacent_same = sum(max(0, run_len - 1) for run_len in script_runs)
            adjacent_ratio = adjacent_same / (count - 1) if count > 1 else 0.0
            if (
                ratio < 0.01
                and count >= self.script_min_suspicious_count
                and avg_run <= 1.4
                and len1_run_ratio >= 0.7
                and adjacent_ratio <= 0.2
            ):
                rare_fragmented_scripts.append(
                    {
                        "script": script,
                        "count": count,
                        "ratio": ratio,
                        "avg_run": avg_run,
                        "len1_run_ratio": len1_run_ratio,
                        "adjacent_ratio": adjacent_ratio,
                    }
                )

        suspicious_counts = {
            script: count
            for script, count in script_counts.items()
            if script not in _SCRIPT_MIX_ALLOWED and count >= self.script_min_suspicious_count
        }
        suspicious_total = sum(suspicious_counts.values())
        suspicious_ratio = suspicious_total / total if total else 0.0
        dense_suspicious_scripts = [
            {
                "script": script,
                "count": count,
                "ratio": count / total if total else 0.0,
            }
            for script, count in suspicious_counts.items()
            if total and count >= self.script_dense_min_count and count / total >= self.script_dense_min_ratio
        ]
        broad_script_mix = (
            total >= self.script_min_chars
            and len(suspicious_counts) >= self.script_min_kinds
            and suspicious_ratio >= self.script_min_ratio
        )
        replacement_count = script_counts.get("Replacement", 0)
        invalid_utf8_mix = replacement_count >= self.script_min_suspicious_count and len(suspicious_counts) >= 2
        script_mix_detected = bool(
            rare_fragmented_scripts or dense_suspicious_scripts or broad_script_mix or invalid_utf8_mix
        )

        return {
            "script_counts": script_counts,
            "suspicious_script_counts": suspicious_counts,
            "suspicious_script_ratio": suspicious_ratio,
            "rare_fragmented_scripts": rare_fragmented_scripts,
            "dense_suspicious_scripts": dense_suspicious_scripts,
            "replacement_count": replacement_count,
            "script_mix_detected": script_mix_detected,
        }

    def _analyze_text_ngram_repetition(self, text: str) -> dict[str, object]:
        tokens = _TEXT_NGRAM_TOKEN_RE.findall(text.lower())
        if len(tokens) < self.text_ngram_size:
            return {
                "text_ngram_size": self.text_ngram_size,
                "text_ngram_max_count": 0,
                "text_ngram_match": None,
            }

        counts = Counter(
            tuple(tokens[index : index + self.text_ngram_size])
            for index in range(len(tokens) - self.text_ngram_size + 1)
        )
        matched_ngram, max_count = counts.most_common(1)[0]
        return {
            "text_ngram_size": self.text_ngram_size,
            "text_ngram_max_count": max_count,
            "text_ngram_match": "".join(matched_ngram),
        }

    def _analyze_reasoning_marker_repetition(self, text: str) -> dict[str, object]:
        lowered = text.lower()
        counts = {marker: lowered.count(marker) for marker in self.reasoning_markers}
        return {
            "reasoning_marker_count": sum(counts.values()),
            "reasoning_marker_counts": counts,
        }

    def _analyze_unclosed_think(self, payload: bytes) -> dict[str, object]:
        prefix = payload[:128].decode("utf-8", errors="replace").lstrip()
        starts_in_think = prefix.startswith(">") or prefix.startswith("<think")
        return {
            "seen_think_close": self._seen_think_close,
            "starts_in_think": starts_in_think,
            "unclosed_think_detected": starts_in_think and not self._seen_think_close,
        }


def repetition_extra_fields(
    *,
    truncated: bool,
    original_response_length: int,
    truncation_length: int | None = None,
    ngram_size: int = DEFAULT_CONSECUTIVE_MAX_PATTERN_SIZE,
    max_count: int = DEFAULT_CONSECUTIVE_MIN_COUNT,
    rules: Sequence[tuple[int, int]] | None = None,
    fallback_max_tokens: int | None = DEFAULT_REPETITION_FALLBACK_MAX_TOKENS,
    matched_rule: tuple[int, int] | None = None,
    matched_reason: str | None = None,
    matched_text_stats: dict[str, object] | None = None,
) -> dict[str, object]:
    matched_text_stats = matched_text_stats or {}
    return {
        "repetition_truncated": truncated,
        "repetition_ngram_size": ngram_size,
        "repetition_ngram_count_threshold": max_count,
        "repetition_fallback_max_tokens": fallback_max_tokens,
        "repetition_detection_rules": (
            [
                {
                    "mode": "consecutive",
                    "min_pattern_size": DEFAULT_CONSECUTIVE_MIN_PATTERN_SIZE,
                    "max_pattern_size": DEFAULT_CONSECUTIVE_MAX_PATTERN_SIZE,
                    "min_count": DEFAULT_CONSECUTIVE_MIN_COUNT,
                }
            ]
            if rules is None
            else [{"ngram_size": rule_ngram_size, "min_count": min_count} for rule_ngram_size, min_count in rules]
        ),
        "repetition_matched_rule": (
            {"ngram_size": matched_rule[0], "min_count": matched_rule[1]} if matched_rule is not None else None
        ),
        "repetition_matched_reason": matched_reason,
        "repetition_truncation_length": truncation_length,
        "original_response_length": original_response_length,
        "repetition_text_byte_length": matched_text_stats.get("text_byte_length"),
        "repetition_zstd_ratio": matched_text_stats.get("zstd_ratio"),
        "repetition_zstd_window_ratio": matched_text_stats.get("zstd_window_ratio"),
        "repetition_script_counts": matched_text_stats.get("script_counts"),
        "repetition_suspicious_script_counts": matched_text_stats.get("suspicious_script_counts"),
        "repetition_suspicious_script_ratio": matched_text_stats.get("suspicious_script_ratio"),
        "repetition_script_window_suspicious_counts": matched_text_stats.get("script_window_suspicious_counts"),
        "repetition_script_window_suspicious_ratio": matched_text_stats.get("script_window_suspicious_ratio"),
        "repetition_rare_fragmented_scripts": matched_text_stats.get("rare_fragmented_scripts"),
        "repetition_dense_suspicious_scripts": matched_text_stats.get("dense_suspicious_scripts"),
        "repetition_script_window_dense_suspicious_scripts": matched_text_stats.get(
            "script_window_dense_suspicious_scripts"
        ),
        "repetition_replacement_count": matched_text_stats.get("replacement_count"),
        "repetition_text_ngram_size": matched_text_stats.get("text_ngram_size"),
        "repetition_text_ngram_max_count": matched_text_stats.get("text_ngram_max_count"),
        "repetition_text_ngram_match": matched_text_stats.get("text_ngram_match"),
        "repetition_reasoning_marker_count": matched_text_stats.get("reasoning_marker_count"),
        "repetition_reasoning_marker_counts": matched_text_stats.get("reasoning_marker_counts"),
        "repetition_seen_think_close": matched_text_stats.get("seen_think_close"),
        "repetition_starts_in_think": matched_text_stats.get("starts_in_think"),
        "repetition_unclosed_think_detected": matched_text_stats.get("unclosed_think_detected"),
    }


def vllm_repetition_detection_config(
    *,
    min_pattern_size: int = DEFAULT_CONSECUTIVE_MIN_PATTERN_SIZE,
    max_pattern_size: int = DEFAULT_CONSECUTIVE_MAX_PATTERN_SIZE,
    min_count: int = DEFAULT_CONSECUTIVE_MIN_COUNT,
) -> dict[str, object]:
    return {
        "max_pattern_size": max_pattern_size,
        "min_pattern_size": min_pattern_size,
        "min_count": min_count,
        "mode": "consecutive",
    }


async def consume_until_repetition(
    stream: AsyncIterable[T],
    *,
    get_token_ids: Callable[[T], Sequence[int]],
    abort_request: Callable[[], Awaitable[object]],
    detector: ConsecutiveRepetitionDetector | NGramRepetitionDetector | None = None,
    cumulative: bool = False,
) -> tuple[T | None, int | None, list[int]]:
    detector = detector or ConsecutiveRepetitionDetector()
    final_output: T | None = None
    observed_token_ids: list[int] = []

    async for output in stream:
        final_output = output
        token_ids = get_token_ids(output)
        if token_ids:
            observed_token_ids = _merge_observed_token_ids(
                observed_token_ids,
                token_ids,
                cumulative=cumulative,
            )

        truncation_length = detector.observe(observed_token_ids)
        if truncation_length is not None:
            await abort_request()
            return final_output, truncation_length, observed_token_ids

    return final_output, None, observed_token_ids


async def consume_token_stream(
    stream: AsyncIterable[T],
    *,
    get_token_ids: Callable[[T], Sequence[int]],
    cumulative: bool = False,
) -> tuple[T | None, list[int]]:
    final_output: T | None = None
    observed_token_ids: list[int] = []

    async for output in stream:
        final_output = output
        token_ids = get_token_ids(output)
        if token_ids:
            observed_token_ids = _merge_observed_token_ids(
                observed_token_ids,
                token_ids,
                cumulative=cumulative,
            )

    return final_output, observed_token_ids


def _merge_observed_token_ids(
    observed_token_ids: list[int],
    token_ids: Sequence[int],
    *,
    cumulative: bool,
) -> list[int]:
    if cumulative:
        if len(token_ids) < len(observed_token_ids):
            observed_token_ids.clear()
            start_index = 0
        else:
            start_index = len(observed_token_ids)
        for index in range(start_index, len(token_ids)):
            observed_token_ids.append(int(token_ids[index]))
        return observed_token_ids

    observed_token_ids.extend(int(token_id) for token_id in token_ids)
    return observed_token_ids
