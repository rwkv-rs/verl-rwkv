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

import torch

RWKV_LM_ROLLOUT_WEIGHT_DTYPE = torch.bfloat16


def _to_rollout_weight_dtype(weight: Any) -> Any:
    if isinstance(weight, torch.Tensor) and torch.is_floating_point(weight):
        return weight.to(dtype=RWKV_LM_ROLLOUT_WEIGHT_DTYPE)
    return weight


def export_rwkv_lm_weights(weights: Iterable[tuple[str, Any]]) -> Generator[tuple[str, Any], None, None]:
    """Export standard Transformers state-dict weights for rollout synchronization."""

    for name, weight in weights:
        yield name, _to_rollout_weight_dtype(weight)


def iter_rwkv_lm_state_dict_weights(model_or_state: Any) -> Generator[tuple[str, Any], None, None]:
    """Iterate native rwkv-lm ``state_dict`` weights without layout changes."""

    if hasattr(model_or_state, "state_dict") and hasattr(model_or_state, "peft_config"):
        state = _snapshot_merged_peft_state_dict(model_or_state)
    else:
        state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    yield from export_rwkv_lm_weights(state.items())


def _snapshot_merged_peft_state_dict(model: Any) -> dict[str, Any]:
    """Snapshot merged PEFT weights and exactly restore the recurrent base."""

    from peft.tuners.lora import LoraLayer

    layers = [module for module in model.modules() if isinstance(module, LoraLayer)]
    backups = [(layer, layer.get_base_layer().weight.detach().clone(), bool(layer.merged)) for layer in layers]
    try:
        for layer in layers:
            if not layer.merged:
                layer.merge(safe_merge=True)
        state = {}
        for name, value in model.state_dict().items():
            normalized_name = name.replace("base_model.model.", "").replace("base_model.", "")
            normalized_name = normalized_name.replace(".base_layer", "")
            if "lora_" in normalized_name or ".adapter_" in normalized_name:
                continue
            state[normalized_name] = value.detach().clone() if isinstance(value, torch.Tensor) else value
        return state
    finally:
        with torch.no_grad():
            for layer, base_weight, was_merged in backups:
                if layer.merged and not was_merged:
                    layer.unmerge()
                layer.get_base_layer().weight.copy_(base_weight)
