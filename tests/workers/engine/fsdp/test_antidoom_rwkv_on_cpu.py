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

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, Rwkv7Config, Rwkv7ForCausalLM

from verl.workers.config import FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def _tiny_rwkv7() -> Rwkv7ForCausalLM:
    return Rwkv7ForCausalLM(
        Rwkv7Config(
            vocab_size=32,
            context_length=16,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            head_size=32,
            wkv_backend="reference",
        )
    )


def test_standard_fsdp_antidoom_peft_train_save_load_merge_and_reload(tmp_path: Path):
    torch.manual_seed(31)
    base_directory = tmp_path / "base"
    _tiny_rwkv7().save_pretrained(base_directory, safe_serialization=True)
    model_config = HFModelConfig(
        path=str(base_directory),
        load_tokenizer=False,
        use_remove_padding=False,
        enable_gradient_checkpointing=False,
        lora={
            "adapter": "antidoom_rwkv",
            "rank": 2,
            "alpha": 4,
            "dropout": 0.0,
            "target_modules": ["receptance", "key", "value", "output"],
        },
    )
    assert model_config.lora_rank == 2
    assert model_config.lora_alpha == 4
    assert model_config.target_modules == ["receptance", "key", "value", "output"]

    engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
    engine.model_config = model_config
    engine.optimizer_config = FSDPOptimizerConfig(lr=0.05, weight_decay=0.0)
    base = AutoModelForCausalLM.from_pretrained(base_directory)
    recurrent_weight_before = base.model.blocks[0].att.receptance.weight.detach().clone()
    low_rank_weight_before = base.model.blocks[0].att.w1.weight.detach().clone()
    model = engine._build_lora_module(base)
    optimizer = engine._build_optimizer(model)

    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all("lora_" in name for name in trainable)
    adapter_before = {name: parameter.detach().clone() for name, parameter in trainable.items()}

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    model.train()
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    assert torch.isfinite(loss)
    assert any(not torch.equal(adapter_before[name], parameter) for name, parameter in trainable.items())
    base_model = model.get_base_model()
    torch.testing.assert_close(base_model.model.blocks[0].att.receptance.base_layer.weight, recurrent_weight_before)
    torch.testing.assert_close(base_model.model.blocks[0].att.w1.weight, low_rank_weight_before)

    model.eval()
    expected = model(input_ids=input_ids).logits.detach()
    adapter_directory = tmp_path / "adapter"
    model.save_pretrained(adapter_directory, safe_serialization=True)
    assert (adapter_directory / "adapter_config.json").is_file()
    assert (adapter_directory / "adapter_model.safetensors").is_file()

    loaded = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_directory),
        adapter_directory,
        is_trainable=True,
    ).eval()
    torch.testing.assert_close(loaded(input_ids=input_ids).logits, expected)

    merged = loaded.merge_and_unload(safe_merge=True).eval()
    torch.testing.assert_close(merged(input_ids=input_ids).logits, expected)
    assert all("lora_" not in name and "base_layer" not in name for name in merged.state_dict())
    merged_directory = tmp_path / "merged"
    merged.save_pretrained(merged_directory, safe_serialization=True)
    assert (merged_directory / "model.safetensors").is_file()

    expected_path = tmp_path / "expected.pt"
    input_path = tmp_path / "input.pt"
    torch.save(expected, expected_path)
    torch.save(input_ids, input_path)
    fresh_process = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_RELOAD, str(merged_directory), str(input_path), str(expected_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert fresh_process.returncode == 0, fresh_process.stderr


_FRESH_PROCESS_RELOAD = r"""
import sys
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(sys.argv[1]).eval()
input_ids = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
expected = torch.load(sys.argv[3], map_location="cpu", weights_only=True)
actual = model(input_ids=input_ids).logits
torch.testing.assert_close(actual, expected)
"""
