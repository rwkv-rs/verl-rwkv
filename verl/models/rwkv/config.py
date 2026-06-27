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

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.models.rwkv.tokenizer import build_rwkv_tokenizer


@dataclass
class RWKVNativeModelConfig(BaseConfig):
    """Shared model configuration template for native RWKV integration.

    This config is intentionally independent from ``HFModelConfig`` because the
    native rwkv-lm path owns model construction, checkpoint naming, precision
    flags, and CUDA extension environment variables.
    """

    _mutable_fields = {"model_type", "tokenizer", "processor"}

    path: str = MISSING
    tokenizer_path: Optional[str] = None
    model_type: str = "language_model"
    load_tokenizer: bool = True
    tokenizer: Any = None
    processor: Any = None
    custom_chat_template: Optional[str] = None
    rwkv_lm_path: Optional[str] = None
    rwkv_version: str = "v7"
    ctx_len: Optional[int] = None
    n_layer: Optional[int] = None
    n_embd: Optional[int] = None
    head_size: Optional[int] = None
    precision: str = "bf16"
    vocab_size: Optional[int] = None
    use_remove_padding: bool = False
    use_fused_kernels: bool = False
    lora: dict[str, object] = field(default_factory=dict)
    native_env: dict[str, str] = field(default_factory=dict)
    weight_mapping: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        assert self.rwkv_version in ["v7"], f"rwkv_version {self.rwkv_version} not supported"
        assert self.precision in ["bf16", "fp16", "fp32"], f"precision {self.precision} not supported"
        if self.load_tokenizer and self.tokenizer is None:
            self.tokenizer = build_rwkv_tokenizer(
                tokenizer_path=self.tokenizer_path,
                pickleable=True,
            )
