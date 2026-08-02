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

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from packaging.requirements import Requirement

from verl.models.transformers import rwkv_runtime

ROOT = Path(__file__).resolve().parents[2]


class _Distribution:
    version = "0.23.1rc1.dev1892+gc97557ccb.rwkv"

    def __init__(self, root: Path, direct_url: dict | None) -> None:
        self.root = root
        self.direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return None if self.direct_url is None else json.dumps(self.direct_url)

    def locate_file(self, filename: str) -> Path:
        return self.root / filename


def _direct_url(revision: str = rwkv_runtime.VLLM_RWKV_REVISION) -> dict:
    return {
        "url": rwkv_runtime.VLLM_RWKV_REPOSITORY,
        "vcs_info": {
            "vcs": "git",
            "requested_revision": revision,
            "commit_id": revision,
        },
    }


def _requirements(path: Path) -> dict[str, Requirement]:
    return {
        requirement.name: requirement
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(("#", "-r "))
        for requirement in (Requirement(line),)
    }


def _model_config(*, model_type: str = "rwkv7", backend: str = "flash_rwkv"):
    config_type = type(
        "Rwkv7Config",
        (),
        {
            "__module__": rwkv_runtime.RWKV7_CONFIG_MODULE,
            "model_type": model_type,
            "architectures": ["Rwkv7ForCausalLM"],
            "wkv_backend": backend,
        },
    )
    return config_type()


def test_rwkv_install_profile_pins_the_self_owned_chain_without_changing_generic_transformers() -> None:
    generic = _requirements(ROOT / "requirements.txt")
    rwkv = _requirements(ROOT / "requirements-rwkv.txt")

    assert "-r requirements-common.txt" in (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "-r requirements-common.txt" in (ROOT / "requirements-rwkv.txt").read_text(encoding="utf-8")
    assert generic["transformers"] == Requirement("transformers>=5.5.3,!=5.6.0,<5.11")
    assert rwkv["vllm"].url == ("git+https://github.com/rwkv-rs/vllm-rwkv.git@c97557ccb1c884a1068edb018dca74ffcde6ec81")
    assert rwkv["transformers"].url == (
        "git+https://github.com/rwkv-rs/transformers-rwkv.git@2696927df9363b5fa175076bb827ba4da2c4e581"
    )
    assert rwkv["flash-linear-attention"].extras == {"flash-rwkv"}
    assert rwkv["flash-linear-attention"].url == (
        "git+https://github.com/rwkv-rs/fla-rwkv.git@a4a8aa98df6ec5322f194a80ec57363dd045adfc"
    )
    assert rwkv["flash-rwkv"].url == (
        "git+https://github.com/rwkv-rs/FlashRWKV.git@866aafd2eed146b0eda1ce03444009ae030f89e3"
    )


def test_registry_vllm_is_rejected_even_if_it_exposes_an_rwkv_name(monkeypatch, tmp_path: Path) -> None:
    distribution = _Distribution(tmp_path, direct_url=None)
    monkeypatch.setattr(rwkv_runtime.importlib_metadata, "distribution", lambda _: distribution)

    with pytest.raises(rwkv_runtime.RwkvRuntimeError, match="rejects registry or ordinary upstream vLLM"):
        rwkv_runtime.validate_rwkv_runtime(tmp_path)


def test_exact_vllm_revision_and_public_chain_are_required(monkeypatch, tmp_path: Path) -> None:
    distribution = _Distribution(tmp_path, direct_url=_direct_url("0" * 40))
    monkeypatch.setattr(rwkv_runtime.importlib_metadata, "distribution", lambda _: distribution)

    with pytest.raises(rwkv_runtime.RwkvRuntimeError, match="vLLM provenance mismatch"):
        rwkv_runtime.validate_rwkv_runtime(tmp_path)


def test_downstream_revision_mismatch_is_rejected() -> None:
    provenance = {
        "revision": rwkv_runtime.TRANSFORMERS_RWKV_REVISION,
        "config_module": rwkv_runtime.RWKV7_CONFIG_MODULE,
        "causal_lm_module": rwkv_runtime.RWKV7_CAUSAL_LM_MODULE,
        "operator_runtime": {
            "revision": "0" * 40,
            "flash_rwkv_revision": rwkv_runtime.FLASH_RWKV_REVISION,
        },
    }

    with pytest.raises(rwkv_runtime.RwkvRuntimeError, match="downstream provenance mismatch"):
        rwkv_runtime._validate_public_chain(provenance)


@pytest.mark.parametrize("backend", ["auto", "reference", "mock", None])
def test_non_flash_provider_is_rejected(backend) -> None:
    with pytest.raises(rwkv_runtime.RwkvRuntimeError, match="auto/reference/mock fallback is disabled"):
        rwkv_runtime._validate_model_config(_model_config(backend=backend))


def test_upstream_rwkv4_identity_is_rejected() -> None:
    with pytest.raises(rwkv_runtime.RwkvRuntimeError, match="upstream RWKV4"):
        rwkv_runtime._validate_model_config(_model_config(model_type="rwkv"))


def test_exact_public_chain_and_flash_artifact_are_accepted(monkeypatch, tmp_path: Path) -> None:
    distribution = _Distribution(tmp_path, direct_url=_direct_url())
    module_path = tmp_path / "vllm/transformers_utils/rwkv7_provenance.py"
    monkeypatch.setattr(rwkv_runtime.importlib_metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(
        rwkv_runtime.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(origin=str(module_path)),
    )

    operator_runtime = {
        "revision": rwkv_runtime.FLA_RWKV_REVISION,
        "flash_rwkv_revision": rwkv_runtime.FLASH_RWKV_REVISION,
    }
    transformers_provenance = {
        "revision": rwkv_runtime.TRANSFORMERS_RWKV_REVISION,
        "config_module": rwkv_runtime.RWKV7_CONFIG_MODULE,
        "causal_lm_module": rwkv_runtime.RWKV7_CAUSAL_LM_MODULE,
        "operator_runtime": operator_runtime,
    }
    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []
    transformers_utils_module = ModuleType("vllm.transformers_utils")
    transformers_utils_module.__path__ = []
    provenance_module = ModuleType("vllm.transformers_utils.rwkv7_provenance")
    provenance_module.validate_transformers_rwkv7_runtime_provenance = lambda: transformers_provenance
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.transformers_utils", transformers_utils_module)
    monkeypatch.setitem(sys.modules, "vllm.transformers_utils.rwkv7_provenance", provenance_module)

    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: _model_config())

    provenance = rwkv_runtime.validate_rwkv_runtime(tmp_path)

    assert provenance["vllm"]["revision"] == rwkv_runtime.VLLM_RWKV_REVISION
    assert provenance["transformers"] == transformers_provenance
    assert provenance["provider"] == "flash_rwkv"


def test_fresh_process_accepts_only_the_pinned_chain_or_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "rwkv7"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Rwkv7ForCausalLM"],
                "model_type": "rwkv7",
                "wkv_backend": "flash_rwkv",
            }
        ),
        encoding="utf-8",
    )
    code = textwrap.dedent(
        f"""
        import json
        from verl.models.transformers.rwkv_runtime import validate_rwkv_runtime

        try:
            provenance = validate_rwkv_runtime({str(checkpoint)!r})
        except RuntimeError as error:
            print(json.dumps({{"status": "rejected", "error": str(error)}}))
        else:
            print(json.dumps({{"status": "accepted", "provenance": provenance}}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout.splitlines()[-1])

    if result["status"] == "rejected":
        assert rwkv_runtime.VLLM_RWKV_REQUIREMENT in result["error"]
        return
    assert result["provenance"]["vllm"]["revision"] == rwkv_runtime.VLLM_RWKV_REVISION
    assert result["provenance"]["transformers"]["revision"] == rwkv_runtime.TRANSFORMERS_RWKV_REVISION
    assert result["provenance"]["provider"] == "flash_rwkv"
