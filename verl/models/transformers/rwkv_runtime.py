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

"""Fail-closed provenance boundary for the self-owned RWKV7 runtime."""

from __future__ import annotations

import importlib.util
import json
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

VLLM_RWKV_DISTRIBUTION = "vllm"
VLLM_RWKV_REPOSITORY = "https://github.com/rwkv-rs/vllm-rwkv.git"
VLLM_RWKV_REVISION = "c97557ccb1c884a1068edb018dca74ffcde6ec81"
VLLM_RWKV_REQUIREMENT = f"vllm @ git+{VLLM_RWKV_REPOSITORY}@{VLLM_RWKV_REVISION}"
TRANSFORMERS_RWKV_REVISION = "2696927df9363b5fa175076bb827ba4da2c4e581"
FLA_RWKV_REVISION = "a4a8aa98df6ec5322f194a80ec57363dd045adfc"
FLASH_RWKV_REVISION = "866aafd2eed146b0eda1ce03444009ae030f89e3"
RWKV7_CONFIG_MODULE = "transformers.models.rwkv7.configuration_rwkv7"
RWKV7_CAUSAL_LM_MODULE = "transformers.models.rwkv7.modeling_rwkv7"


class RwkvRuntimeError(RuntimeError):
    """The installed runtime cannot satisfy the RWKV7 product contract."""


def _vllm_distribution_provenance() -> dict[str, str]:
    try:
        distribution = importlib_metadata.distribution(VLLM_RWKV_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError as error:
        raise RwkvRuntimeError(f"MaxRL RWKV7 requires `{VLLM_RWKV_REQUIREMENT}`.") from error

    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RwkvRuntimeError(
            f"MaxRL RWKV7 rejects registry or ordinary upstream vLLM; install `{VLLM_RWKV_REQUIREMENT}`."
        )
    try:
        direct_url = json.loads(direct_url_text)
    except (TypeError, ValueError) as error:
        raise RwkvRuntimeError(
            f"MaxRL RWKV7 vLLM direct_url.json is invalid; reinstall `{VLLM_RWKV_REQUIREMENT}`."
        ) from error

    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict) or vcs_info.get("vcs") != "git":
        raise RwkvRuntimeError(f"MaxRL RWKV7 requires a pinned Git vLLM install; install `{VLLM_RWKV_REQUIREMENT}`.")
    repository = str(direct_url.get("url", "")).removeprefix("git+").rstrip("/")
    requested_revision = str(vcs_info.get("requested_revision", "")).lower()
    resolved_revision = str(vcs_info.get("commit_id", "")).lower()
    if repository != VLLM_RWKV_REPOSITORY or (
        requested_revision != VLLM_RWKV_REVISION or resolved_revision != VLLM_RWKV_REVISION
    ):
        raise RwkvRuntimeError(
            "MaxRL RWKV7 vLLM provenance mismatch: "
            f"repository={repository!r}, requested={requested_revision!r}, resolved={resolved_revision!r}; "
            f"install `{VLLM_RWKV_REQUIREMENT}`."
        )

    module_name = "vllm.transformers_utils.rwkv7_provenance"
    try:
        module_spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError) as error:
        raise RwkvRuntimeError("The pinned vllm-rwkv public provenance module is not importable") from error
    expected_origin = Path(distribution.locate_file("vllm/transformers_utils/rwkv7_provenance.py")).resolve()
    if module_spec is None or module_spec.origin is None or Path(module_spec.origin).resolve() != expected_origin:
        raise RwkvRuntimeError(
            "MaxRL RWKV7 public provenance module does not belong to the pinned vllm-rwkv distribution"
        )
    return {
        "distribution": VLLM_RWKV_DISTRIBUTION,
        "distribution_version": distribution.version,
        "repository": VLLM_RWKV_REPOSITORY,
        "revision": VLLM_RWKV_REVISION,
    }


def _validate_public_chain(provenance: Any) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise RwkvRuntimeError("vllm-rwkv returned an invalid Transformers-RWKV provenance contract")
    operator_runtime = provenance.get("operator_runtime")
    expected = {
        "revision": TRANSFORMERS_RWKV_REVISION,
        "config_module": RWKV7_CONFIG_MODULE,
        "causal_lm_module": RWKV7_CAUSAL_LM_MODULE,
        "operator_runtime.revision": FLA_RWKV_REVISION,
        "operator_runtime.flash_rwkv_revision": FLASH_RWKV_REVISION,
    }
    actual = {
        "revision": provenance.get("revision"),
        "config_module": provenance.get("config_module"),
        "causal_lm_module": provenance.get("causal_lm_module"),
        "operator_runtime.revision": operator_runtime.get("revision") if isinstance(operator_runtime, dict) else None,
        "operator_runtime.flash_rwkv_revision": (
            operator_runtime.get("flash_rwkv_revision") if isinstance(operator_runtime, dict) else None
        ),
    }
    mismatches = {name: (expected[name], actual[name]) for name in expected if actual[name] != expected[name]}
    if mismatches:
        raise RwkvRuntimeError(f"MaxRL RWKV7 downstream provenance mismatch: {mismatches}")
    return provenance


def _validate_model_config(model_config: Any) -> None:
    model_type = str(getattr(model_config, "model_type", "")).lower()
    architectures = tuple(getattr(model_config, "architectures", None) or ())
    if (
        model_type != "rwkv7"
        or type(model_config).__module__ != RWKV7_CONFIG_MODULE
        or "Rwkv7ForCausalLM" not in architectures
    ):
        raise RwkvRuntimeError(
            "MaxRL only accepts the self-owned RWKV7 HF artifact; upstream RWKV4 and unrelated models are rejected"
        )
    if getattr(model_config, "wkv_backend", None) != "flash_rwkv":
        raise RwkvRuntimeError(
            "MaxRL RWKV7 requires wkv_backend='flash_rwkv'; auto/reference/mock fallback is disabled"
        )


def validate_rwkv_runtime(checkpoint: str | Path) -> dict[str, Any]:
    """Validate exact vLLM/Transformers/operator ownership and the HF artifact."""

    vllm_provenance = _vllm_distribution_provenance()
    try:
        from vllm.transformers_utils.rwkv7_provenance import (
            validate_transformers_rwkv7_runtime_provenance,
        )
    except ImportError as error:
        raise RwkvRuntimeError(
            "The pinned vllm-rwkv install does not expose its public RWKV7 provenance contract"
        ) from error

    try:
        transformers_provenance = _validate_public_chain(validate_transformers_rwkv7_runtime_provenance())
    except RwkvRuntimeError:
        raise
    except (ImportError, RuntimeError) as error:
        raise RwkvRuntimeError(f"MaxRL RWKV7 downstream runtime rejected provenance: {error}") from error

    from transformers import AutoConfig

    try:
        model_config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RwkvRuntimeError(
            f"MaxRL RWKV7 cannot load the HF artifact config at {str(checkpoint)!r}: {error}"
        ) from error
    _validate_model_config(model_config)
    return {
        "vllm": vllm_provenance,
        "transformers": transformers_provenance,
        "provider": "flash_rwkv",
    }


__all__ = [
    "FLASH_RWKV_REVISION",
    "FLA_RWKV_REVISION",
    "RWKV7_CAUSAL_LM_MODULE",
    "RWKV7_CONFIG_MODULE",
    "RwkvRuntimeError",
    "TRANSFORMERS_RWKV_REVISION",
    "VLLM_RWKV_REQUIREMENT",
    "VLLM_RWKV_REVISION",
    "validate_rwkv_runtime",
]
