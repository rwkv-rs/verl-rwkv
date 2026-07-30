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
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from .paths import resolve_rwkv_lm_paths

_IMPORT_LOCK = threading.RLock()
_active_rwkv_lm_root: Path | None = None
_active_native_env: tuple[tuple[str, str], ...] | None = None


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
    previous = list(sys.path)
    sys.path[:] = [path_str, *(item for item in previous if item != path_str)]
    try:
        yield
    finally:
        sys.path[:] = previous


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _module_paths(module: ModuleType) -> tuple[Path, ...]:
    file = getattr(module, "__file__", None)
    if file is not None:
        return (Path(file).resolve(),)
    return tuple(Path(path).resolve() for path in getattr(module, "__path__", ()))


def _is_from_checkout(module: ModuleType, source_dir: Path) -> bool:
    paths = _module_paths(module)
    return bool(paths) and all(path.is_relative_to(source_dir) for path in paths)


def _require_module_origin(
    module: ModuleType,
    *,
    module_name: str,
    expected_path: Path,
) -> None:
    paths = _module_paths(module)
    if paths != (expected_path.resolve(),):
        rendered = ", ".join(map(str, paths)) or "<unknown>"
        raise RuntimeError(
            f"native rwkv-lm module {module_name!r} came from {rendered}; expected {expected_path.resolve()}."
        )


def _reject_foreign_src_modules(source_dir: Path) -> None:
    for module_name in ("src", "src.model", "src.trainer"):
        module = sys.modules.get(module_name)
        if module is not None and not _is_from_checkout(module, source_dir):
            rendered = ", ".join(map(str, _module_paths(module))) or "<unknown>"
            raise RuntimeError(
                f"cannot import native rwkv-lm because {module_name!r} is "
                f"already loaded from {rendered}; expected {source_dir}."
            )


def _rollback_new_checkout_modules(
    previous_module_names: set[str],
) -> None:
    for module_name in tuple(sys.modules):
        if module_name not in previous_module_names and (module_name == "src" or module_name.startswith("src.")):
            sys.modules.pop(module_name, None)


def _native_env_fingerprint(native_env: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in (native_env or {}).items()))


def import_rwkv_lm_modules(
    *,
    rwkv_lm_path: str | None = None,
    native_env: Mapping[str, str] | None = None,
) -> tuple[ModuleType, ModuleType]:
    """Atomically import native model and trainer modules from one checkout.

    The legacy flat-layout checkout imports as the generic ``src`` package and
    reads configuration and CUDA source paths from process-global state during
    import. Consequently, a worker process may bind exactly one canonical
    checkout. The successful modules intentionally remain in ``sys.modules`` so
    pickle, Lightning, and lazy imports retain stable module identities.
    """

    global _active_native_env, _active_rwkv_lm_root

    paths = resolve_rwkv_lm_paths(rwkv_lm_path)
    root = paths.repo_root.resolve()
    source_dir = root / "src"
    native_env_fingerprint = _native_env_fingerprint(native_env)

    with _IMPORT_LOCK:
        if _active_rwkv_lm_root is not None and _active_rwkv_lm_root != root:
            raise RuntimeError(
                "native rwkv-lm already uses checkout "
                f"{_active_rwkv_lm_root}; one checkout per worker process is "
                f"supported, cannot switch to {root}."
            )
        if _active_native_env is not None and _active_native_env != native_env_fingerprint:
            raise RuntimeError(
                "native rwkv-lm is already imported with different "
                "import-time environment values in this worker process."
            )

        _reject_foreign_src_modules(source_dir)
        if _active_rwkv_lm_root == root:
            model_module = sys.modules.get("src.model")
            trainer_module = sys.modules.get("src.trainer")
            if model_module is None or trainer_module is None:
                raise RuntimeError("native rwkv-lm module cache was modified after import; restart the worker process.")
            _require_module_origin(
                model_module,
                module_name="src.model",
                expected_path=source_dir / "model.py",
            )
            _require_module_origin(
                trainer_module,
                module_name="src.trainer",
                expected_path=source_dir / "trainer.py",
            )
            return model_module, trainer_module

        preloaded_children = {module_name for module_name in ("src.model", "src.trainer") if module_name in sys.modules}
        if preloaded_children:
            raise RuntimeError(
                "cannot adopt preloaded native rwkv-lm modules without "
                "verified import-time environment provenance: " + ", ".join(sorted(preloaded_children))
            )

        previous_module_names = set(sys.modules)
        importlib.invalidate_caches()
        try:
            with (
                _patched_env(native_env),
                _prepended_sys_path(root),
                _working_directory(root),
            ):
                model_module = importlib.import_module("src.model")
                trainer_module = importlib.import_module("src.trainer")
            _require_module_origin(
                model_module,
                module_name="src.model",
                expected_path=source_dir / "model.py",
            )
            _require_module_origin(
                trainer_module,
                module_name="src.trainer",
                expected_path=source_dir / "trainer.py",
            )
        except BaseException:
            _rollback_new_checkout_modules(previous_module_names)
            raise

        _active_rwkv_lm_root = root
        _active_native_env = native_env_fingerprint
        return model_module, trainer_module
