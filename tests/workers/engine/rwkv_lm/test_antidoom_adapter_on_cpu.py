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

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

from verl.workers.engine.rwkv_lm.antidoom_adapter import (
    ANTIDOOM_RWKV_ADAPTER,
    AntidoomRWKVAdapterConfig,
    antidoom_ftpo_loss,
    antidoom_trainable_parameters,
    apply_configured_antidoom_rwkv_adapter,
    evaluate_antidoom_rwkv,
    inject_antidoom_rwkv_adapter,
    load_antidoom_rwkv_adapter,
    merge_antidoom_rwkv_adapter,
    restrict_optimizer_to_antidoom_adapter,
    save_antidoom_rwkv_adapter,
    save_merged_rwkv_checkpoint,
)
from verl.workers.engine.rwkv_lm.weight_bridge import iter_rwkv_lm_state_dict_weights


class _ToyTimeMix(nn.Module):
    """Small tensor path with the same recurrent and native low-rank names as RWKV7."""

    def __init__(self, width: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(width, 2) * 0.1)
        self.w2 = nn.Parameter(torch.randn(2, width) * 0.1)
        self.a1 = nn.Parameter(torch.randn(width, 2) * 0.1)
        self.a2 = nn.Parameter(torch.randn(2, width) * 0.1)
        self.v1 = nn.Parameter(torch.randn(width, 2) * 0.1)
        self.v2 = nn.Parameter(torch.randn(2, width) * 0.1)
        self.g1 = nn.Parameter(torch.randn(width, 2) * 0.1)
        self.g2 = nn.Parameter(torch.randn(2, width) * 0.1)
        self.r_k = nn.Parameter(torch.randn(1, width) * 0.1)
        self.receptance = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        mixed = torch.tanh(hidden @ self.w1) @ self.w2
        recurrent = self.receptance(hidden) + self.key(hidden) + self.value(hidden)
        return hidden + self.output(torch.tanh(recurrent + mixed))


class _ToyBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.att = _ToyTimeMix(width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.att(hidden)


class _ToyRWKV(nn.Module):
    def __init__(self, vocabulary: int = 8, width: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocabulary, width)
        self.blocks = nn.ModuleList([_ToyBlock(width)])
        self.head = nn.Linear(width, vocabulary, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.emb(token_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)


class _ToyNativeRWKV(_ToyRWKV):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.trainer = None

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _new_base(state: dict[str, torch.Tensor] | None = None) -> _ToyRWKV:
    model = _ToyRWKV()
    if state is not None:
        model.load_state_dict(state)
    return model


def _train_real_adapter_step():
    torch.manual_seed(17)
    base = _new_base()
    base_state = {name: value.detach().clone() for name, value in base.state_dict().items()}
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    reference_logits = base(input_ids)[:, -1, :].detach()
    recurrent_weight_before = base.blocks[0].att.receptance.weight.detach().clone()
    native_low_rank_before = base.blocks[0].att.w1.detach().clone()

    model = inject_antidoom_rwkv_adapter(
        base,
        AntidoomRWKVAdapterConfig(rank=2, alpha=4, dropout=0.0),
    )
    trainable = antidoom_trainable_parameters(model)
    adapter_before = {name: parameter.detach().clone() for name, parameter in trainable}

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    retained = restrict_optimizer_to_antidoom_adapter(optimizer, model)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimizer_ids == {id(parameter) for _, parameter in trainable}
    assert retained == sum(parameter.numel() for _, parameter in trainable)

    logits = model(input_ids)[:, -1, :]
    loss, metrics = antidoom_ftpo_loss(
        logits,
        chosen_token_ids=torch.tensor([[1, 2], [2, 3]]),
        rejected_token_ids=torch.tensor([6, 7]),
        reference_logits=reference_logits,
    )
    loss.backward()
    optimizer.step()

    base_model = model.get_base_model()
    torch.testing.assert_close(base_model.blocks[0].att.receptance.base_layer.weight, recurrent_weight_before)
    torch.testing.assert_close(base_model.blocks[0].att.w1, native_low_rank_before)
    assert any(
        not torch.equal(parameter, adapter_before[name])
        for name, parameter in antidoom_trainable_parameters(model)
        if ".lora_B." in name
    )
    assert 0.0 <= metrics["chosen_win"].item() <= 1.0
    return model, base_state, input_ids


def test_antidoom_adapter_trains_only_recurrent_lora_tensors_and_exposes_eval():
    model, _, input_ids = _train_real_adapter_step()

    trainable_names = [name for name, _ in antidoom_trainable_parameters(model)]
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)
    assert all(any(target in name for target in ("receptance", "key", "value", "output")) for name in trainable_names)

    metrics = evaluate_antidoom_rwkv(
        model,
        input_ids,
        chosen_token_ids=torch.tensor([[1, 2], [2, 3]]),
        rejected_token_ids=torch.tensor([6, 7]),
    )
    assert set(metrics) == {
        "pref_loss",
        "chosen_win",
        "margin_win",
        "mean_delta",
        "median_delta",
        "active_weight",
        "mse_elem",
        "mse_tgt_tokenwise",
    }
    assert model.training


def test_antidoom_adapter_save_load_merge_and_fresh_process_reload(tmp_path: Path):
    model, base_state, input_ids = _train_real_adapter_step()
    expected = model(input_ids).detach()

    adapter_directory = save_antidoom_rwkv_adapter(model, tmp_path / "adapter")
    assert {path.name for path in adapter_directory.iterdir()} >= {
        "adapter_config.json",
        "adapter_model.safetensors",
    }

    loaded = load_antidoom_rwkv_adapter(_new_base(base_state), adapter_directory)
    torch.testing.assert_close(loaded(input_ids), expected)

    merged = merge_antidoom_rwkv_adapter(_new_base(base_state), adapter_directory)
    torch.testing.assert_close(merged(input_ids), expected)
    merged_checkpoint = save_merged_rwkv_checkpoint(merged, tmp_path / "merged.pth")
    assert all("lora_" not in name and "base_layer" not in name for name in merged.state_dict())

    expected_path = tmp_path / "expected.pt"
    input_path = tmp_path / "input.pt"
    torch.save(expected, expected_path)
    torch.save(input_ids, input_path)
    fresh_process = subprocess.run(
        [
            sys.executable,
            "-c",
            _FRESH_PROCESS_RELOAD,
            str(merged_checkpoint),
            str(input_path),
            str(expected_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert fresh_process.returncode == 0, fresh_process.stderr


def test_configured_adapter_and_rollout_export_use_merged_recurrent_weights():
    torch.manual_seed(9)
    base = _new_base()
    model = apply_configured_antidoom_rwkv_adapter(
        base,
        SimpleNamespace(
            lora={
                "adapter": ANTIDOOM_RWKV_ADAPTER,
                "rank": 2,
                "alpha": 4,
                "target_modules": ["receptance", "key", "value", "output"],
            }
        ),
    )
    for name, parameter in antidoom_trainable_parameters(model):
        if ".lora_B." in name:
            parameter.data.fill_(0.25)

    base_layer = model.get_base_model().blocks[0].att.receptance
    unmerged_weight = base_layer.base_layer.weight.detach().clone()
    expected_merged_weight = unmerged_weight + base_layer.get_delta_weight("default")
    exported = dict(iter_rwkv_lm_state_dict_weights(model))

    torch.testing.assert_close(
        exported["blocks.0.att.receptance.weight"],
        expected_merged_weight.to(torch.bfloat16),
    )
    torch.testing.assert_close(base_layer.base_layer.weight, unmerged_weight)
    assert not base_layer.merged
    assert all("lora_" not in name and "base_layer" not in name for name in exported)


def test_native_runner_engine_injects_adapter_and_filters_native_optimizer():
    from verl.workers.config import RWKVLMEngineConfig, RWKVLMOptimizerConfig
    from verl.workers.engine.rwkv_lm import RWKVLMEngine
    from verl.workers.engine.rwkv_lm.native_runner import NativeRWKVLMRunner

    model_module = ModuleType("src.model")
    model_module.RWKV = _ToyNativeRWKV
    trainer_module = ModuleType("src.trainer")
    trainer_module.train_callback = lambda args: SimpleNamespace(args=args)

    def importer(module_name, **_):
        return model_module if module_name == "src.model" else trainer_module

    class ToyRunner(NativeRWKVLMRunner):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, importer=importer)

    model_config = SimpleNamespace(
        path="",
        lora={
            "adapter": ANTIDOOM_RWKV_ADAPTER,
            "rank": 2,
            "alpha": 4,
        },
    )
    engine = RWKVLMEngine(
        model_config=model_config,
        engine_config=RWKVLMEngineConfig(rwkv_lm_path="/unused", param_offload=True),
        optimizer_config=RWKVLMOptimizerConfig(lr=0.1),
        checkpoint_config=None,
        runner_cls=ToyRunner,
    ).initialize()

    assert hasattr(engine.model, "peft_config")
    optimizer_parameters = [parameter for group in engine.optimizer.param_groups for parameter in group["params"]]
    assert optimizer_parameters
    assert all(parameter.requires_grad for parameter in optimizer_parameters)
    assert {id(parameter) for parameter in optimizer_parameters} == {
        id(parameter) for _, parameter in antidoom_trainable_parameters(engine.model)
    }
    assert engine.model.get_base_model().trainer is not None
    assert engine.disable_adapter() is not None


_FRESH_PROCESS_RELOAD = r"""
import sys
import torch
from torch import nn

class TimeMix(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(width, 2))
        self.w2 = nn.Parameter(torch.empty(2, width))
        self.a1 = nn.Parameter(torch.empty(width, 2))
        self.a2 = nn.Parameter(torch.empty(2, width))
        self.v1 = nn.Parameter(torch.empty(width, 2))
        self.v2 = nn.Parameter(torch.empty(2, width))
        self.g1 = nn.Parameter(torch.empty(width, 2))
        self.g2 = nn.Parameter(torch.empty(2, width))
        self.r_k = nn.Parameter(torch.empty(1, width))
        self.receptance = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
    def forward(self, hidden):
        mixed = torch.tanh(hidden @ self.w1) @ self.w2
        recurrent = self.receptance(hidden) + self.key(hidden) + self.value(hidden)
        return hidden + self.output(torch.tanh(recurrent + mixed))

class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.att = TimeMix(width)
    def forward(self, hidden):
        return self.att(hidden)

class Model(nn.Module):
    def __init__(self, vocabulary=8, width=4):
        super().__init__()
        self.emb = nn.Embedding(vocabulary, width)
        self.blocks = nn.ModuleList([Block(width)])
        self.head = nn.Linear(width, vocabulary, bias=False)
    def forward(self, token_ids):
        hidden = self.emb(token_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)

model = Model()
model.load_state_dict(torch.load(sys.argv[1], map_location="cpu", weights_only=True))
actual = model(torch.load(sys.argv[2], map_location="cpu", weights_only=True))
expected = torch.load(sys.argv[3], map_location="cpu", weights_only=True)
torch.testing.assert_close(actual, expected)
"""
