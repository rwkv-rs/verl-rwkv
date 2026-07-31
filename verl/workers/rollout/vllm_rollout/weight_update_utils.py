# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import logging
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

WeightUpdate = tuple[str, torch.Tensor]
logger = logging.getLogger(__name__)


def _disable_mtp_completeness_check():
    """Use vLLM's scoped MTP control when the installed version provides it."""
    try:
        from vllm.model_executor.model_loader.mtp_validation import (
            disable_mtp_completeness_check,
        )
    except ImportError:
        return nullcontext()
    return disable_mtp_completeness_check()


@dataclass
class _ReloadTarget:
    model: torch.nn.Module
    model_config: Any
    named_buffers: dict[str, torch.Tensor]
    expected: set[str] | None
    loaded: set[str] | None
    needs_finalize: bool = False


class _LayerwiseReloadSession:
    """Own one complete vLLM layerwise reload across all received buckets."""

    def __init__(
        self,
        models_with_config: Iterable[tuple[torch.nn.Module, Any]],
        *,
        initialize: Callable[[torch.nn.Module], None],
        finalize: Callable[[torch.nn.Module, Any], None],
    ) -> None:
        self._targets: list[_ReloadTarget] = []
        self._finalize = finalize
        self._staged_buffers: list[WeightUpdate] = []
        self._finished = False

        for model, model_config in models_with_config:
            strict = getattr(model_config, "quantization", None) is None
            self._targets.append(
                _ReloadTarget(
                    model=model,
                    model_config=model_config,
                    named_buffers=dict(model.named_buffers()),
                    expected={name for name, _ in model.named_parameters()} if strict else None,
                    loaded=set() if strict else None,
                )
            )

        try:
            for target in self._targets:
                target.needs_finalize = True
                initialize(target.model)
        except Exception:
            self.abort()
            raise

    def load_bucket(self, weights: list[WeightUpdate]) -> None:
        if self._finished:
            raise RuntimeError("Cannot load weights after the reload session finished")
        if not self._targets:
            return

        owned_weights = [(name, tensor.detach().clone()) for name, tensor in weights]
        main_buffers = self._targets[0].named_buffers
        param_updates = []
        for name, tensor in owned_weights:
            if name in main_buffers:
                self._staged_buffers.append((name, tensor))
            else:
                param_updates.append((name, tensor))

        if not param_updates:
            return
        for target in self._targets:
            with _disable_mtp_completeness_check():
                loaded = target.model.load_weights(param_updates)
            if loaded is None:
                target.loaded = None
            elif target.loaded is not None:
                target.loaded.update(loaded)

    def finish(self) -> None:
        if self._finished:
            raise RuntimeError("Reload session already finished")

        self._finalize_models()
        for target in self._targets:
            apply_buffer_updates(
                target.model,
                self._staged_buffers,
                named_buffers=target.named_buffers,
            )
            if (
                target.expected is not None
                and target.loaded is not None
                and (missing := target.expected - target.loaded)
            ):
                logger.warning(
                    "Following weights were not loaded from checkpoint: %s",
                    missing,
                )
        self._finished = True

    def abort(self) -> None:
        if self._finished:
            return
        try:
            self._finalize_models()
        except Exception:
            logger.exception("Failed to finalize an interrupted vLLM weight reload")
        finally:
            self._staged_buffers.clear()
            self._finished = True

    def _finalize_models(self) -> None:
        first_error = None
        for target in self._targets:
            if not target.needs_finalize:
                continue
            try:
                self._finalize(target.model, target.model_config)
            except Exception as error:
                if first_error is None:
                    first_error = error
            finally:
                target.needs_finalize = False
        if first_error is not None:
            raise first_error


def split_buffer_updates(
    model: torch.nn.Module, weights: list[WeightUpdate]
) -> tuple[list[WeightUpdate], list[WeightUpdate], dict[str, torch.Tensor]]:
    """Split incoming weight updates into parameter and buffer updates.

    Returns the parameter updates, the buffer updates, and the model's
    ``named_buffers`` map so callers can reuse it without re-iterating.
    """
    named_buffers = dict(model.named_buffers())
    param_updates, buffer_updates = [], []
    for name, tensor in weights:
        if name in named_buffers:
            buffer_updates.append((name, tensor))
        else:
            param_updates.append((name, tensor))
    return param_updates, buffer_updates, named_buffers


@torch.no_grad()
def apply_buffer_updates(
    model: torch.nn.Module,
    buffer_updates: list[WeightUpdate],
    named_buffers: dict[str, torch.Tensor] | None = None,
) -> int:
    """Copy updated buffer tensors into the target model in-place."""
    if not buffer_updates:
        return 0

    if named_buffers is None:
        named_buffers = dict(model.named_buffers())
    loaded = 0
    for name, tensor in buffer_updates:
        if name not in named_buffers:
            continue

        target = named_buffers[name]
        if target.shape != tensor.shape:
            raise ValueError(
                f"Buffer shape mismatch for {name}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}"
            )

        source = tensor.to(device=target.device, dtype=target.dtype, non_blocking=False)
        target.copy_(source, non_blocking=False)
        loaded += 1

    return loaded
