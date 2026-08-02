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

import importlib.util

import torch


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeModel:
    def state_dict(self):
        return {
            "model.embeddings.weight": "emb",
            "head.weight": "head",
        }


def test_rwkv_lm_weight_bridge_preserves_standard_hf_state_dict_keys():
    bridge = _load_module("rwkv_lm_weight_bridge_test", "verl/workers/engine/rwkv_lm/weight_bridge.py")

    assert list(bridge.iter_rwkv_lm_state_dict_weights(FakeModel())) == [
        ("model.embeddings.weight", "emb"),
        ("head.weight", "head"),
    ]


def test_rwkv_lm_weight_bridge_exports_floating_tensors_as_bf16():
    bridge = _load_module("rwkv_lm_weight_bridge_dtype_test", "verl/workers/engine/rwkv_lm/weight_bridge.py")

    weights = dict(
        bridge.export_rwkv_lm_weights([("model.embeddings.weight", torch.ones(2, dtype=torch.float32))])
    )

    assert weights["model.embeddings.weight"].dtype is torch.bfloat16
