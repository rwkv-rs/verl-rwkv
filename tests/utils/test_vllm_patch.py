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
from pathlib import Path

PATCH_MODULE = Path(__file__).resolve().parents[2] / "verl" / "utils" / "vllm" / "patch.py"
spec = importlib.util.spec_from_file_location("verl_vllm_patch", PATCH_MODULE)
patch = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(patch)


def test_moe_weight_loader_patch_ignores_direct_non_moe_model(monkeypatch):
    class SupportedMoeModel:
        pass

    class DirectNonMoeModel:
        pass

    monkeypatch.setattr(patch, "SUPPORTED_MOE_MODELS", [SupportedMoeModel])

    patch.patch_vllm_moe_model_weight_loader(DirectNonMoeModel())
