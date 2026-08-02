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

__all__ = [
    "PPOTrainer",
    "register_trainer",
    "get_trainer_cls",
    "PPOTrainerSync",
    "PPOTrainerColocateAsync",
    "PPOTrainerSeparateAsync",
    "AgentLoopWorkerTQ",
    "AgentLoopManagerTQ",
]

_EXPORT_MODULES = {
    "PPOTrainer": ".trainer_base",
    "register_trainer": ".trainer_base",
    "get_trainer_cls": ".trainer_base",
    "PPOTrainerSync": ".trainer_sync",
    "PPOTrainerColocateAsync": ".trainer_colocate_async",
    "PPOTrainerSeparateAsync": ".trainer_separate_async",
    "AgentLoopWorkerTQ": ".agent_loop_tq",
    "AgentLoopManagerTQ": ".agent_loop_tq",
}


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(name)
    from importlib import import_module

    value = getattr(import_module(_EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value
