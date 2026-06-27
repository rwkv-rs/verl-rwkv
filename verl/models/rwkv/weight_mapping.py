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

from collections.abc import Generator, Iterable
from typing import Any

VERL_TO_RWKV_LM_WEIGHT_MAP: dict[str, str] = {}
RWKV_LM_DEEPSPEED_PREFIX = "_forward_module."


def strip_rwkv_lm_deepspeed_prefix(name: str) -> str:
    """Strip rwkv-lm's DeepSpeed wrapper prefix.

    Source: ``rwkv-lm/RWKV-v7/train_temp/train.py`` rewrites
    ``_forward_module.*`` checkpoint keys immediately after ``torch.load``.
    """

    if name.startswith(RWKV_LM_DEEPSPEED_PREFIX):
        return name.replace(RWKV_LM_DEEPSPEED_PREFIX, "", 1)
    return name


def _map_name(name: str, explicit_map: dict[str, str]) -> str:
    normalized = strip_rwkv_lm_deepspeed_prefix(name)
    return explicit_map.get(normalized, normalized)


def map_verl_to_rwkv_lm(weights: Iterable[tuple[str, Any]]) -> Generator[tuple[str, Any], None, None]:
    """Map Verl trainer weight names into native rwkv-lm names."""

    for name, weight in weights:
        yield _map_name(name, VERL_TO_RWKV_LM_WEIGHT_MAP), weight
