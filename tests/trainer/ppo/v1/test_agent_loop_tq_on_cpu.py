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

import asyncio

import torch
from tensordict import TensorDict

from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ


class _AsyncRemoteMethod:
    def __init__(self, worker_id: int, completions: list[tuple[int, int]]) -> None:
        self.worker_id = worker_id
        self.completions = completions

    def remote(self, chunk: TensorDict):
        async def complete() -> None:
            await asyncio.sleep(0)
            self.completions.append((self.worker_id, len(chunk)))

        return complete()


class _Worker:
    def __init__(self, worker_id: int, completions: list[tuple[int, int]]) -> None:
        self.generate_sequences = _AsyncRemoteMethod(worker_id, completions)


def test_tq_dispatch_awaits_worker_acknowledgements_without_ray_get():
    completions: list[tuple[int, int]] = []
    manager = AgentLoopManagerTQ.__new__(AgentLoopManagerTQ)
    manager.agent_loop_workers = [_Worker(0, completions), _Worker(1, completions)]
    prompts = TensorDict({"input_ids": torch.arange(8).reshape(4, 2)}, batch_size=[4])

    manager.generate_sequences(prompts)

    assert sorted(completions) == [(0, 2), (1, 2)]
