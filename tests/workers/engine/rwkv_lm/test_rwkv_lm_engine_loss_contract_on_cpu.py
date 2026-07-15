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
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu


def test_rwkv_lm_engine_populates_verl_loss_global_batch_fields():
    from verl.workers.config import RWKVLMEngineConfig, RWKVLMOptimizerConfig
    from verl.workers.engine.rwkv_lm import RWKVLMEngine

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.seen_input_shape = None

        def forward(self, input_ids):
            self.seen_input_shape = tuple(input_ids.shape)
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, 8, dtype=torch.float32)
            return logits + self.weight

    engine = RWKVLMEngine(
        model_config=None,
        engine_config=RWKVLMEngineConfig(),
        optimizer_config=RWKVLMOptimizerConfig(),
        checkpoint_config=None,
    )
    engine.model = FakeModel()

    data = TensorDict(
        {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "responses": torch.tensor([[2, 3], [5, 6]]),
            "loss_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.float32),
        },
        batch_size=[2],
    )

    def loss_function(model_output, data, dp_group):
        assert data["batch_num_tokens"] == 3
        assert data["dp_size"] == 1
        assert data["global_batch_size"] == 2
        return model_output["log_probs"].sum() * 0 + engine.model.weight, {}

    engine.forward_backward_batch(data, loss_function=loss_function)

    assert engine.model.seen_input_shape == (2, 16)
    assert engine.model.weight.grad is not None


def test_rwkv_lm_engine_padded_inference_returns_temperature_scaled_entropy():
    from types import SimpleNamespace

    from verl.workers.config import RWKVLMEngineConfig, RWKVLMOptimizerConfig
    from verl.workers.engine.rwkv_lm import RWKVLMEngine

    engine = RWKVLMEngine(
        model_config=None,
        engine_config=RWKVLMEngineConfig(),
        optimizer_config=RWKVLMOptimizerConfig(),
        checkpoint_config=None,
    )
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [3.0, 1.0, -2.0],
                [2.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    )
    responses = torch.tensor([[0, 1]])
    data = TensorDict({}, batch_size=[1])
    tu.assign_non_tensor_data(data, "calculate_entropy", True)
    tu.assign_non_tensor_data(data, "temperature", 2.0)

    output = engine._build_model_output(
        logits,
        SimpleNamespace(input_ids=torch.tensor([[4, 5, 0, 1]]), responses=responses),
        data,
    )

    expected_logits = logits[:, 1:3] / 2.0
    expected_entropy = torch.distributions.Categorical(logits=expected_logits).entropy()
    assert torch.allclose(output["entropy"], expected_entropy)
    assert output["log_probs"].shape == responses.shape


def test_rwkv_lm_engine_uses_infctx_sequence_forward_when_enabled():
    from verl.workers.config import RWKVLMEngineConfig, RWKVLMOptimizerConfig
    from verl.workers.engine.rwkv_lm import RWKVLMEngine

    class FakeInfctxModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.seen_chunks = []
            self.seen_direct_forward = False

        def forward(self, input_ids):
            self.seen_direct_forward = True
            raise AssertionError("infctx engine path must not call direct forward")

        def forward_infctx_sequence(self, input_ids, *, chunk_ctx):
            for start in range(0, input_ids.size(1), chunk_ctx):
                self.seen_chunks.append(tuple(input_ids[:, start : start + chunk_ctx].shape))
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, 8, dtype=torch.float32)
            return logits + self.weight

    engine = RWKVLMEngine(
        model_config=None,
        engine_config=RWKVLMEngineConfig(infctx=True, chunk_ctx=32),
        optimizer_config=RWKVLMOptimizerConfig(),
        checkpoint_config=None,
    )
    engine.model = FakeInfctxModel()

    data = TensorDict(
        {
            "input_ids": torch.arange(68).view(2, 34),
            "responses": torch.tensor([[1, 2], [3, 4]]),
            "loss_mask": torch.ones(2, 2, dtype=torch.float32),
        },
        batch_size=[2],
    )

    def loss_function(model_output, data, dp_group):
        return model_output["log_probs"].sum() * 0 + engine.model.weight, {}

    engine.forward_backward_batch(data, loss_function=loss_function)

    assert engine.model.seen_direct_forward is False
    assert engine.model.seen_chunks == [(2, 32), (2, 16)]
    assert engine.model.weight.grad is not None


def test_rwkv_lm_engine_infctx_chunks_response_log_probs_and_detaches_state():
    from verl.workers.config import RWKVLMEngineConfig, RWKVLMOptimizerConfig
    from verl.workers.engine.rwkv_lm import RWKVLMEngine

    class FakeChunkInfctxModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.emb = torch.nn.Embedding(8, 4)
            self.head = torch.nn.Linear(4, 8, bias=False)
            self.seen_chunks = []

        def forward_infctx_sequence(self, input_ids, *, chunk_ctx):
            raise AssertionError("chunked infctx response path must not build full-sequence logits")

        def create_infctx_state(self, batch_size, device, dtype):
            shift_states = torch.zeros(1, 2, batch_size, 1, device=device, dtype=dtype)
            wkv_states = torch.zeros(1, batch_size, 1, 1, 1, device=device, dtype=dtype)
            return shift_states, wkv_states

        def forward_infctx_chunk(self, input_ids, shift_states, wkv_states):
            self.seen_chunks.append(
                {
                    "shape": tuple(input_ids.shape),
                    "grad_enabled": torch.is_grad_enabled(),
                    "shift_requires_grad": shift_states.requires_grad,
                    "wkv_requires_grad": wkv_states.requires_grad,
                }
            )
            hidden = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).to(dtype=torch.float32)
            hidden = hidden + self.weight
            state_value = hidden[:, -1, :1].mean()
            new_shift_states = shift_states + state_value
            new_wkv_states = wkv_states + state_value.reshape(1, 1, 1, 1, 1)
            return hidden, new_shift_states, new_wkv_states

    engine = RWKVLMEngine(
        model_config=None,
        engine_config=RWKVLMEngineConfig(infctx=True, chunk_ctx=16, ctx_len=80),
        optimizer_config=RWKVLMOptimizerConfig(),
        checkpoint_config=None,
    )
    engine.model = FakeChunkInfctxModel()

    data = TensorDict(
        {
            "input_ids": (torch.arange(70).view(1, 70) % 8),
            "responses": (torch.arange(40).view(1, 40) % 8),
            "loss_mask": torch.ones(1, 40, dtype=torch.float32),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor_data(data, "calculate_entropy", True)
    tu.assign_non_tensor_data(data, "temperature", 2.0)

    def loss_function(model_output, data, dp_group):
        assert tuple(model_output["log_probs"].shape) == (1, 40)
        assert tuple(model_output["entropy"].shape) == (1, 40)
        assert torch.isfinite(model_output["entropy"]).all()
        return -model_output["log_probs"].sum(), {}

    engine.forward_backward_batch(data, loss_function=loss_function)

    assert [chunk["shape"] for chunk in engine.model.seen_chunks] == [(1, 16)] * 5
    assert [chunk["grad_enabled"] for chunk in engine.model.seen_chunks] == [False, True, True, True, True]
    assert all(not chunk["shift_requires_grad"] for chunk in engine.model.seen_chunks)
    assert all(not chunk["wkv_requires_grad"] for chunk in engine.model.seen_chunks)
    assert engine.model.weight.grad is not None
    assert engine.model.head.weight.grad is not None


def test_rwkv_lm_engine_infctx_chunked_output_satisfies_no_padding_loss_contract():
    from verl.workers.config import RWKVLMEngineConfig, RWKVLMOptimizerConfig
    from verl.workers.engine.rwkv_lm import RWKVLMEngine
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

    class FakeChunkInfctxModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.bfloat16))
            self.emb = torch.nn.Embedding(32, 4, dtype=torch.bfloat16)
            self.head = torch.nn.Linear(4, 32, bias=False, dtype=torch.bfloat16)

        def forward_infctx_sequence(self, input_ids, *, chunk_ctx):
            raise AssertionError("chunked infctx response path must not build full-sequence logits")

        def create_infctx_state(self, batch_size, device, dtype):
            shift_states = torch.zeros(1, 2, batch_size, 1, device=device, dtype=dtype)
            wkv_states = torch.zeros(1, batch_size, 1, 1, 1, device=device, dtype=dtype)
            return shift_states, wkv_states

        def forward_infctx_chunk(self, input_ids, shift_states, wkv_states):
            hidden = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).to(dtype=self.weight.dtype)
            hidden = hidden + self.weight
            state_value = hidden[:, -1, :1].mean()
            new_shift_states = shift_states + state_value
            new_wkv_states = wkv_states + state_value.reshape(1, 1, 1, 1, 1)
            return hidden, new_shift_states, new_wkv_states

    engine = RWKVLMEngine(
        model_config=None,
        engine_config=RWKVLMEngineConfig(infctx=True, chunk_ctx=16, ctx_len=32),
        optimizer_config=RWKVLMOptimizerConfig(),
        checkpoint_config=None,
    )
    engine.model = FakeChunkInfctxModel()

    data = TensorDict(
        {
            "prompts": torch.tensor(
                [
                    [10, 11, 12, 13, 14, 15, 16, 17],
                    [0, 0, 0, 0, 0, 0, 20, 21],
                ]
            ),
            "responses": torch.tensor(
                [
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [9, 10, 0, 0, 0, 0, 0, 0],
                ]
            ),
            "input_ids": torch.tensor(
                [
                    [10, 11, 12, 13, 14, 15, 16, 17, 1, 2, 3, 4, 5, 6, 7, 8],
                    [0, 0, 0, 0, 0, 0, 20, 21, 9, 10, 0, 0, 0, 0, 0, 0],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                ]
            ),
            "response_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                ],
                dtype=torch.float32,
            ),
            "position_ids": torch.arange(16).repeat(2, 1),
        },
        batch_size=[2],
    )
    data = left_right_2_no_padding(data)
    tu.assign_non_tensor_data(data, "calculate_entropy", True)
    tu.assign_non_tensor_data(data, "temperature", 2.0)

    def loss_function(model_output, data, dp_group):
        log_probs = no_padding_2_padding(model_output["log_probs"], data)
        entropy = no_padding_2_padding(model_output["entropy"], data)
        assert tuple(log_probs.shape) == (2, 8)
        assert tuple(entropy.shape) == (2, 8)
        # Behavior-policy probabilities use the same FP32 log-softmax
        # reduction as vLLM even when model logits are BF16.
        assert log_probs.dtype == torch.float32
        assert torch.isfinite(entropy[data["response_mask"].bool()]).all()
        return -log_probs[data["response_mask"].bool()].sum(), {}

    engine.forward_backward_batch(data, loss_function=loss_function)

    assert engine.model.weight.grad is not None
    assert engine.model.head.weight.grad is not None


def test_rwkv_lm_engine_honors_static_micro_batch_size():
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric import Metric
    from verl.workers.config import RWKVLMEngineConfig, RWKVLMOptimizerConfig
    from verl.workers.engine.rwkv_lm import RWKVLMEngine

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.seen_input_shapes = []

        def forward(self, input_ids):
            self.seen_input_shapes.append(tuple(input_ids.shape))
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, 8, dtype=torch.float32)
            return logits + self.weight

    engine = RWKVLMEngine(
        model_config=None,
        engine_config=RWKVLMEngineConfig(),
        optimizer_config=RWKVLMOptimizerConfig(),
        checkpoint_config=None,
    )
    engine.model = FakeModel()

    data = TensorDict(
        {
            "input_ids": torch.tensor(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [1, 3, 5],
                    [2, 4, 6],
                ]
            ),
            "responses": torch.tensor(
                [
                    [2, 3],
                    [5, 6],
                    [3, 5],
                    [4, 6],
                ]
            ),
            "loss_mask": torch.ones(4, 2, dtype=torch.float32),
        },
        batch_size=[4],
    )
    tu.assign_non_tensor(data, use_dynamic_bsz=False, micro_batch_size_per_gpu=1)

    loss_batch_sizes = []

    def loss_function(model_output, data, dp_group):
        loss_batch_sizes.append(data.batch_size[0])
        assert data["batch_num_tokens"] == 8
        assert data["global_batch_size"] == 4
        return model_output["log_probs"].sum() * 0 + engine.model.weight, {
            "loss_batch_size": data.batch_size[0],
            "metric_batch_size": Metric("mean", data.batch_size[0]),
        }

    output = engine.forward_backward_batch(data, loss_function=loss_function)

    assert engine.model.seen_input_shapes == [(1, 16)] * 4
    assert loss_batch_sizes == [1, 1, 1, 1]
    assert output["loss"] == [1.0, 1.0, 1.0, 1.0]
    assert output["metrics"]["loss_batch_size"] == [1, 1, 1, 1]
    assert output["metrics"]["actual_micro_batches"] == [4]
    assert isinstance(output["metrics"]["metric_batch_size"], Metric)
    assert output["metrics"]["metric_batch_size"].aggregate() == 1.0
    assert engine.model.weight.grad.item() == 4.0


def test_rwkv_lm_engine_restores_dynamic_micro_batch_order_from_index_lists():
    from verl.workers.config import RWKVLMEngineConfig
    from verl.workers.engine.rwkv_lm import RWKVLMEngine

    engine = RWKVLMEngine(
        model_config=None,
        engine_config=RWKVLMEngineConfig(),
        optimizer_config=None,
        checkpoint_config=None,
    )
    output_lst = [
        {
            "model_output": {
                "log_probs": torch.tensor([[20.0], [30.0]]),
                "labels": ["two", "three"],
            }
        },
        {
            "model_output": {
                "log_probs": torch.tensor([[0.0], [10.0]]),
                "labels": ["zero", "one"],
            }
        },
    ]

    merged = engine._merge_micro_batch_model_outputs(
        output_lst,
        indices=[[2, 3], [0, 1]],
        data=TensorDict({}, batch_size=[4]),
    )

    torch.testing.assert_close(merged["log_probs"], torch.tensor([[0.0], [10.0], [20.0], [30.0]]))
    assert merged["labels"] == ["zero", "one", "two", "three"]


def test_rwkv_lm_engine_averages_gradients_before_optimizer_step(monkeypatch):
    from verl.workers.config import RWKVLMEngineConfig, RWKVLMOptimizerConfig
    from verl.workers.engine.rwkv_lm import RWKVLMEngine, transformer_impl

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(10.0)
    model.weight.grad = torch.full_like(model.weight, 2.0)
    all_reduce_calls = []

    def fake_all_reduce(tensor, op=None, group=None):
        all_reduce_calls.append((tensor, op, group))
        tensor.add_(8.0)

    fake_group = object()
    monkeypatch.setattr(transformer_impl.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(transformer_impl.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(transformer_impl.torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(transformer_impl.torch.distributed, "all_reduce", fake_all_reduce)

    engine = RWKVLMEngine(
        model_config=None,
        engine_config=RWKVLMEngineConfig(),
        optimizer_config=RWKVLMOptimizerConfig(clip_grad=100.0),
        checkpoint_config=None,
    )
    engine.model = model
    engine.optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    engine.get_data_parallel_group = lambda: fake_group

    grad_norm = engine.optimizer_step()

    assert len(all_reduce_calls) == 1
    assert all_reduce_calls[0][1] is transformer_impl.torch.distributed.ReduceOp.SUM
    assert all_reduce_calls[0][2] is fake_group
    assert model.weight.grad is None
    torch.testing.assert_close(model.weight, torch.tensor([[5.0]]))
    torch.testing.assert_close(grad_norm, torch.tensor(5.0))
