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
"""Path contracts for native RWKV upstream checkouts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


RWKV_LM_ENV = "RWKV_LM_PATH"

RWKV_LM_TRAIN_REL = Path("RWKV-v7/train_temp")
RWKV_LM_REQUIRED_FILES = (
    RWKV_LM_TRAIN_REL / "train.py",
    RWKV_LM_TRAIN_REL / "src/model.py",
    RWKV_LM_TRAIN_REL / "src/trainer.py",
)


@dataclass(frozen=True)
class RWKVLMPaths:
    """Resolved paths for a native rwkv-lm checkout."""

    repo_root: Path
    train_dir: Path


def _repo_default(relative: str) -> Path:
    return Path(__file__).resolve().parents[3] / relative


def _resolve_root(path: Optional[str], env_name: str, default_relative: str) -> Path:
    value = path or os.environ.get(env_name)
    root = Path(value).expanduser() if value else _repo_default(default_relative)
    return root.resolve()


def _require_files(root: Path, relative_files: tuple[Path, ...], project_name: str) -> None:
    missing = [str(root / relative) for relative in relative_files if not (root / relative).is_file()]
    if missing:
        missing_list = "\n  - ".join(missing)
        raise FileNotFoundError(f"{project_name} checkout is missing required files:\n  - {missing_list}")


def resolve_rwkv_lm_paths(path: Optional[str] = None, *, require: bool = True) -> RWKVLMPaths:
    """Resolve a native rwkv-lm repository path.

    ``path`` may point at the rwkv-lm repository root. When omitted, this checks
    ``RWKV_LM_PATH`` and then the future repo-local ``third_party/rwkv-lm``.
    """

    repo_root = _resolve_root(path, RWKV_LM_ENV, "third_party/rwkv-lm")
    if require:
        _require_files(repo_root, RWKV_LM_REQUIRED_FILES, "rwkv-lm")
    return RWKVLMPaths(repo_root=repo_root, train_dir=repo_root / RWKV_LM_TRAIN_REL)
