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

from typing import Any

__all__ = [
    "RWKVLMPaths",
    "RWKVNativeModelConfig",
    "build_rwkv_tokenizer",
    "import_rwkv_lm",
    "map_verl_to_rwkv_lm",
    "resolve_rwkv_lm_paths",
]


def __getattr__(name: str) -> Any:
    """Keep model-only utilities importable without loading the vLLM tokenizer."""

    if name == "RWKVNativeModelConfig":
        from .config import RWKVNativeModelConfig

        return RWKVNativeModelConfig
    if name == "import_rwkv_lm":
        from .native_imports import import_rwkv_lm

        return import_rwkv_lm
    if name in {"RWKVLMPaths", "resolve_rwkv_lm_paths"}:
        from .paths import RWKVLMPaths, resolve_rwkv_lm_paths

        return {"RWKVLMPaths": RWKVLMPaths, "resolve_rwkv_lm_paths": resolve_rwkv_lm_paths}[name]
    if name == "build_rwkv_tokenizer":
        from .tokenizer import build_rwkv_tokenizer

        return build_rwkv_tokenizer
    if name == "map_verl_to_rwkv_lm":
        from .weight_mapping import map_verl_to_rwkv_lm

        return map_verl_to_rwkv_lm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
