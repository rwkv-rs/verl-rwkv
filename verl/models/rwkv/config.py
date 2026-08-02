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
    """Checkpoint and architecture identity for native RWKV models.

    Native checkout, runtime, precision, and environment settings belong to
    ``RWKVLMEngineConfig``. This config remains independent from
    ``HFModelConfig`` because native RWKV checkpoints do not use the Hugging
    Face model-loading contract.
    """

    _mutable_fields = {"model_type", "tokenizer", "processor"}

    path: str = MISSING
    repository: Optional[str] = None
    revision: Optional[str] = None
    filename: Optional[str] = None
    sha256: Optional[str] = None
    tokenizer_path: Optional[str] = None
    model_type: str = "language_model"
    load_tokenizer: bool = True
    tokenizer: Any = None
    processor: Any = None
    custom_chat_template: Optional[str] = None
    rwkv_version: str = "v7"
    n_layer: Optional[int] = None
    n_embd: Optional[int] = None
    head_size: Optional[int] = None
    vocab_size: Optional[int] = None
    lora: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        assert self.rwkv_version in ["v7"], f"rwkv_version {self.rwkv_version} not supported"
        if self.load_tokenizer and self.tokenizer is None:
            self.tokenizer = build_rwkv_tokenizer(
                tokenizer_path=self.tokenizer_path,
                pickleable=True,
            )
