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

import pytest
import torch

import verl.models.transformers.monkey_patch as monkey_patch


class _ModelWithoutAttentionConfig(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="rwkv7")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class _AttentionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="synthetic",
            num_attention_heads=8,
            num_key_value_heads=2,
        )


def test_dense_model_does_not_require_attention_config(monkeypatch) -> None:
    monkeypatch.setattr(monkey_patch, "is_trl_available", lambda: False)
    model = _ModelWithoutAttentionConfig()
    original_forward = model.__class__.forward
    inputs = torch.randn(2, 3)

    monkey_patch.apply_monkey_patch(
        model,
        ulysses_sp_size=1,
        use_remove_padding=False,
        use_fused_kernels=False,
    )

    assert model.__class__.forward is original_forward
    torch.testing.assert_close(model(inputs), inputs)


def test_attention_parallelism_still_validates_head_divisibility(
    monkeypatch,
) -> None:
    monkeypatch.setattr(monkey_patch, "is_trl_available", lambda: False)

    with pytest.raises(
        AssertionError,
        match="num_attention_heads 8 must be divisible by ulysses_sp_size 3",
    ):
        monkey_patch.apply_monkey_patch(
            _AttentionModel(),
            ulysses_sp_size=3,
            use_remove_padding=True,
            use_fused_kernels=False,
        )
