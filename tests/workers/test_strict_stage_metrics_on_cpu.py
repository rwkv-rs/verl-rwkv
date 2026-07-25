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

from contextlib import nullcontext

from tensordict import TensorDict

from verl.workers.engine.base import BaseEngine


class _Engine(BaseEngine):
    def __init__(self):
        self.calls = []
        self.last_optimizer_timing = {"gradient_communication": 0.25, "optimizer": 0.5}

    @property
    def is_param_offload_enabled(self):
        return False

    @property
    def is_optimizer_offload_enabled(self):
        return False

    def train_mode(self, **kwargs):
        return nullcontext()

    def eval_mode(self, **kwargs):
        return nullcontext()

    def optimizer_zero_grad(self):
        self.calls.append("zero_grad")

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        self.calls.append("forward_backward")
        return {"metrics": {}, "model_output": {}}

    def optimizer_step(self):
        self.calls.append("optimizer")
        return 1.0

    def is_mp_src_rank_with_outputs(self):
        return True


def test_base_engine_emits_stage_metrics_and_steps_optimizer_once():
    engine = _Engine()
    output = engine.train_batch(TensorDict({}, batch_size=[]), lambda **_: None)

    assert engine.calls == ["zero_grad", "forward_backward", "optimizer"]
    assert output["metrics"]["grad_norm"] == 1.0
    assert output["metrics"]["timing/gradient_communication"] == 0.25
    assert output["metrics"]["timing/optimizer"] == 0.5
    assert output["metrics"]["timing/actor_forward_backward"] >= 0
