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

"""Import helpers for native RWKV upstream checkouts."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from .paths import resolve_rwkv_lm_paths


@contextmanager
def _patched_env(env: Mapping[str, str] | None) -> Iterator[None]:
    if not env:
        yield
        return

    old_values: dict[str, str | None] = {key: os.environ.get(key) for key in env}
    os.environ.update({key: str(value) for key, value in env.items()})
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


@contextmanager
def _prepended_sys_path(path: Path) -> Iterator[None]:
    path_str = str(path)
    added = path_str not in sys.path
    if added:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _import_from_path(
    module_name: str,
    python_path: Path,
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> ModuleType:
    importlib.invalidate_caches()
    with _patched_env(env), _prepended_sys_path(python_path):
        if cwd is None:
            return importlib.import_module(module_name)
        with _working_directory(cwd):
            return importlib.import_module(module_name)


def import_rwkv_lm(
    module_name: str = "src.model",
    *,
    rwkv_lm_path: str | None = None,
    native_env: Mapping[str, str] | None = None,
) -> ModuleType:
    """Import a module from a caller-provided rwkv-lm checkout.

    This is interface glue only. It puts the flattened native training directory
    on ``sys.path`` and imports the requested native module from that checkout.
    Callers importing ``src.model`` must pass the same ``RWKV_*`` environment
    values that rwkv-lm's ``train.py`` would set before import.
    """

    paths = resolve_rwkv_lm_paths(rwkv_lm_path)
    return _import_from_path(module_name, paths.train_dir, cwd=paths.train_dir, env=native_env)
