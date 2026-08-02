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

import torch
from transformers import AutoModelForCausalLM, Rwkv7Config, Rwkv7ForCausalLM

from verl.utils.model import convert_weight_keys


def _tiny_rwkv7() -> Rwkv7ForCausalLM:
    return Rwkv7ForCausalLM(
        Rwkv7Config(
            vocab_size=32,
            context_length=16,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            head_size=32,
            wkv_backend="reference",
        )
    )


def test_rwkv7_hf_artifact_save_load_and_standard_state_dict_sync(tmp_path):
    torch.manual_seed(7)
    actor = _tiny_rwkv7().eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    expected_logits = actor(input_ids=input_ids).logits.detach()

    actor.save_pretrained(tmp_path, safe_serialization=True)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()

    assert isinstance(reloaded, Rwkv7ForCausalLM)
    assert (tmp_path / "model.safetensors").is_file()
    assert set(reloaded.state_dict()) == set(actor.state_dict())
    assert torch.equal(reloaded(input_ids=input_ids).logits, expected_logits)

    rollout = _tiny_rwkv7().eval()
    exported = convert_weight_keys(actor.state_dict(), actor)
    assert "model.embeddings.weight" in exported
    assert "model.blocks.0.att.key.weight" in exported
    assert "emb.weight" not in exported
    rollout.load_state_dict(exported, strict=True)

    assert torch.equal(rollout(input_ids=input_ids).logits, expected_logits)
    for name, actor_tensor in actor.state_dict().items():
        assert torch.equal(rollout.state_dict()[name], actor_tensor), name
