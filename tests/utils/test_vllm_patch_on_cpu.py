# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

from verl.utils.vllm import patch


class DenseModel:
    pass


class SupportedMoeModel:
    def __init__(self):
        self.model = SimpleNamespace(layers=[SimpleNamespace(mlp=Mlp())])


class Mlp:
    def __init__(self):
        self.experts = SimpleNamespace(weight_loader=object())
        self.w13_weight = SimpleNamespace()
        self.w2_weight = SimpleNamespace()
        self.other_weight = SimpleNamespace()

    def named_parameters(self):
        return (
            ("w13_weight", self.w13_weight),
            ("w2_weight", self.w2_weight),
            ("other_weight", self.other_weight),
        )


def test_dense_model_without_wrapper_is_ignored(monkeypatch):
    monkeypatch.setattr(patch, "SUPPORTED_MOE_MODELS", [SupportedMoeModel])

    patch.patch_vllm_moe_model_weight_loader(DenseModel())


def test_wrapped_dense_model_is_ignored(monkeypatch):
    monkeypatch.setattr(patch, "SUPPORTED_MOE_MODELS", [SupportedMoeModel])
    wrapper = SimpleNamespace(model=DenseModel())

    patch.patch_vllm_moe_model_weight_loader(wrapper)


def test_supported_moe_model_patches_only_expert_weights(monkeypatch):
    monkeypatch.setattr(patch, "SUPPORTED_MOE_MODELS", [SupportedMoeModel])
    model = SupportedMoeModel()
    mlp = model.model.layers[0].mlp

    patch.patch_vllm_moe_model_weight_loader(model)

    assert mlp.w13_weight.weight_loader is mlp.experts.weight_loader
    assert mlp.w2_weight.weight_loader is mlp.experts.weight_loader
    assert not hasattr(mlp.other_weight, "weight_loader")
