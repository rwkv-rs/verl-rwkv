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

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest

from verl.models.rwkv import (
    import_rwkv_lm_modules,
    native_imports,
    resolve_rwkv_lm_paths,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _clear_modules(*module_names: str) -> None:
    for module_name in module_names:
        sys.modules.pop(module_name, None)


@pytest.fixture(autouse=True)
def _isolated_src_modules(monkeypatch):
    saved = {name: module for name, module in sys.modules.items() if name == "src" or name.startswith("src.")}
    for name in saved:
        sys.modules.pop(name, None)
    monkeypatch.setattr(native_imports, "_active_rwkv_lm_root", None)
    monkeypatch.setattr(native_imports, "_active_native_env", None)
    yield
    for name in tuple(sys.modules):
        if name == "src" or name.startswith("src."):
            sys.modules.pop(name, None)
    sys.modules.update(saved)


def _write_required_train_files(train_dir: Path) -> None:
    _write(train_dir / "train.py", "")
    _write(train_dir / "src/model.py", "")
    _write(train_dir / "src/trainer.py", "")


def test_resolve_rwkv_lm_paths_accepts_flat_train_layout(tmp_path):
    _write_required_train_files(tmp_path)

    paths = resolve_rwkv_lm_paths(str(tmp_path))

    assert paths.repo_root == tmp_path.resolve()
    assert paths.train_dir == tmp_path.resolve()


def test_resolve_rwkv_lm_paths_rejects_missing_native_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="rwkv-lm checkout is missing required flat-layout files"):
        resolve_rwkv_lm_paths(str(tmp_path))


def test_import_rwkv_lm_modules_uses_flat_train_sys_path_and_cwd(tmp_path):
    _write_required_train_files(tmp_path)
    _write(
        tmp_path / "src/model.py",
        "from pathlib import Path\nCWD = Path.cwd()\nVALUE = 'rwkv-lm'\n",
    )
    previous_cwd = Path.cwd()
    previous_sys_path = list(sys.path)

    model_module, trainer_module = import_rwkv_lm_modules(rwkv_lm_path=str(tmp_path))

    assert model_module.VALUE == "rwkv-lm"
    assert model_module.CWD == tmp_path.resolve()
    assert trainer_module.__file__ == str(tmp_path / "src/trainer.py")
    assert Path.cwd() == previous_cwd
    assert sys.path == previous_sys_path


def test_import_rwkv_lm_modules_patches_native_env_only_during_import(
    tmp_path,
    monkeypatch,
):
    _write_required_train_files(tmp_path)
    _write(
        tmp_path / "src/model.py",
        "import os\nVALUE = os.environ['RWKV_HEAD_SIZE']\n",
    )
    monkeypatch.delenv("RWKV_HEAD_SIZE", raising=False)

    model_module, _ = import_rwkv_lm_modules(
        rwkv_lm_path=str(tmp_path),
        native_env={"RWKV_HEAD_SIZE": "64"},
    )

    assert model_module.VALUE == "64"
    assert "RWKV_HEAD_SIZE" not in os.environ


def test_import_rwkv_lm_modules_reuses_same_checkout(tmp_path):
    _write_required_train_files(tmp_path)
    _write(
        tmp_path / "src/model.py",
        "EXECUTIONS = globals().get('EXECUTIONS', 0) + 1\n",
    )

    first = import_rwkv_lm_modules(rwkv_lm_path=str(tmp_path))
    second = import_rwkv_lm_modules(rwkv_lm_path=str(tmp_path))

    assert second == first
    assert first[0].EXECUTIONS == 1


def test_import_rwkv_lm_modules_requires_same_native_env(tmp_path):
    _write_required_train_files(tmp_path)

    first = import_rwkv_lm_modules(
        rwkv_lm_path=str(tmp_path),
        native_env={"RWKV_HEAD_SIZE": "64", "RWKV_JIT_ON": "0"},
    )
    second = import_rwkv_lm_modules(
        rwkv_lm_path=str(tmp_path),
        native_env={"RWKV_JIT_ON": "0", "RWKV_HEAD_SIZE": "64"},
    )

    assert second == first
    with pytest.raises(RuntimeError, match="different import-time environment"):
        import_rwkv_lm_modules(
            rwkv_lm_path=str(tmp_path),
            native_env={"RWKV_HEAD_SIZE": "128", "RWKV_JIT_ON": "0"},
        )


def test_import_rwkv_lm_modules_rejects_foreign_src_package(tmp_path):
    _write_required_train_files(tmp_path)
    foreign = ModuleType("src")
    foreign.__path__ = [str(tmp_path / "foreign/src")]
    sys.modules["src"] = foreign

    with pytest.raises(RuntimeError, match="already loaded"):
        import_rwkv_lm_modules(rwkv_lm_path=str(tmp_path))

    assert sys.modules["src"] is foreign


def test_import_rwkv_lm_modules_rejects_preloaded_child_without_provenance(
    tmp_path,
):
    _write_required_train_files(tmp_path)
    package = ModuleType("src")
    package.__path__ = [str(tmp_path / "src")]
    model = ModuleType("src.model")
    model.__file__ = str(tmp_path / "src/model.py")
    sys.modules["src"] = package
    sys.modules["src.model"] = model

    with pytest.raises(RuntimeError, match="cannot adopt preloaded"):
        import_rwkv_lm_modules(rwkv_lm_path=str(tmp_path))


def test_import_rwkv_lm_modules_rejects_second_checkout(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_required_train_files(first_root)
    _write_required_train_files(second_root)

    import_rwkv_lm_modules(rwkv_lm_path=str(first_root))

    with pytest.raises(RuntimeError, match="one checkout per worker process"):
        import_rwkv_lm_modules(rwkv_lm_path=str(second_root))


def test_import_rwkv_lm_modules_prioritizes_target_and_restores_sys_path(
    tmp_path,
):
    target_root = tmp_path / "target"
    foreign_root = tmp_path / "foreign"
    _write_required_train_files(target_root)
    _write_required_train_files(foreign_root)
    _write(target_root / "src/model.py", "VALUE = 'target'\n")
    _write(foreign_root / "src/model.py", "VALUE = 'foreign'\n")
    previous_sys_path = list(sys.path)
    sys.path[:0] = [str(foreign_root), str(target_root)]
    configured_sys_path = list(sys.path)
    try:
        model_module, _ = import_rwkv_lm_modules(rwkv_lm_path=str(target_root))
        assert model_module.VALUE == "target"
        assert sys.path == configured_sys_path
    finally:
        sys.path[:] = previous_sys_path


def test_import_rwkv_lm_modules_rolls_back_partial_import(tmp_path):
    _write_required_train_files(tmp_path)
    _write(tmp_path / "src/trainer.py", "raise RuntimeError('broken trainer')\n")
    previous_cwd = Path.cwd()
    previous_sys_path = list(sys.path)

    with pytest.raises(RuntimeError, match="broken trainer"):
        import_rwkv_lm_modules(rwkv_lm_path=str(tmp_path))

    assert native_imports._active_rwkv_lm_root is None
    assert "src.model" not in sys.modules
    assert Path.cwd() == previous_cwd
    assert sys.path == previous_sys_path

    _write(tmp_path / "src/trainer.py", "VALUE = 'fixed'\n")
    _, trainer_module = import_rwkv_lm_modules(rwkv_lm_path=str(tmp_path))
    assert trainer_module.VALUE == "fixed"


def test_import_rwkv_lm_modules_concurrent_calls_import_once(tmp_path):
    _write_required_train_files(tmp_path)
    _write(
        tmp_path / "src/model.py",
        "EXECUTIONS = globals().get('EXECUTIONS', 0) + 1\n",
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _: import_rwkv_lm_modules(rwkv_lm_path=str(tmp_path)),
                range(2),
            )
        )

    assert results[0] == results[1]
    assert results[0][0].EXECUTIONS == 1


def test_import_rwkv_lm_modules_reports_damaged_module_cache(tmp_path):
    _write_required_train_files(tmp_path)
    import_rwkv_lm_modules(rwkv_lm_path=str(tmp_path))
    sys.modules.pop("src.model")

    with pytest.raises(RuntimeError, match="module cache was modified"):
        import_rwkv_lm_modules(rwkv_lm_path=str(tmp_path))
