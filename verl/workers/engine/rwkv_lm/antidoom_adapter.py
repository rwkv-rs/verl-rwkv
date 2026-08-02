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
"""PEFT-backed Antidoom adapter support for native recurrent RWKV models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

ANTIDOOM_RWKV_ADAPTER = "antidoom_rwkv"
DEFAULT_ANTIDOOM_RWKV_TARGET_MODULES = ("receptance", "key", "value", "output")


@dataclass(frozen=True)
class AntidoomRWKVAdapterConfig:
    """Configuration for the standard LoRA adapter used by Antidoom FTPO."""

    rank: int = 128
    alpha: int = 128
    dropout: float = 0.0
    target_modules: tuple[str, ...] = DEFAULT_ANTIDOOM_RWKV_TARGET_MODULES

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("Antidoom RWKV adapter rank must be positive")
        if self.alpha <= 0:
            raise ValueError("Antidoom RWKV adapter alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Antidoom RWKV adapter dropout must be in [0, 1)")
        if not self.target_modules or any(not name.strip() for name in self.target_modules):
            raise ValueError("Antidoom RWKV adapter target_modules must be non-empty names")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AntidoomRWKVAdapterConfig:
        """Build from the native model ``lora`` mapping used by Hydra configs."""

        target_modules = value.get("target_modules", DEFAULT_ANTIDOOM_RWKV_TARGET_MODULES)
        if isinstance(target_modules, str):
            target_modules = tuple(item.strip() for item in target_modules.split(",") if item.strip())
        else:
            target_modules = tuple(str(item) for item in target_modules)
        return cls(
            rank=int(value.get("rank", value.get("r", 128))),
            alpha=int(value.get("alpha", value.get("lora_alpha", 128))),
            dropout=float(value.get("dropout", value.get("lora_dropout", 0.0))),
            target_modules=target_modules,
        )


def configured_antidoom_rwkv_adapter(model_config: Any) -> Mapping[str, Any] | None:
    """Return an explicitly selected Antidoom adapter mapping, if configured."""

    value = getattr(model_config, "lora", None)
    if not isinstance(value, Mapping) or value.get("adapter") != ANTIDOOM_RWKV_ADAPTER:
        return None
    return value


def inject_antidoom_rwkv_adapter(
    model: torch.nn.Module,
    config: AntidoomRWKVAdapterConfig,
) -> torch.nn.Module:
    """Inject PEFT LoRA layers and fail closed unless every target was found."""

    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraLayer

    adapter_model = get_peft_model(
        model,
        LoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=list(config.target_modules),
            bias="none",
        ),
    )
    injected_names = [name for name, module in adapter_model.named_modules() if isinstance(module, LoraLayer)]
    missing_targets = [
        target for target in config.target_modules if not any(name.endswith(f".{target}") for name in injected_names)
    ]
    if missing_targets:
        raise RuntimeError(
            "Antidoom RWKV adapter did not inject every configured target; missing=" + ",".join(missing_targets)
        )
    _validate_antidoom_trainable_parameters(adapter_model)
    return adapter_model


def load_antidoom_rwkv_adapter(
    base_model: torch.nn.Module,
    adapter_directory: str | Path,
    *,
    is_trainable: bool = True,
) -> torch.nn.Module:
    """Load a standard PEFT adapter into a freshly constructed RWKV base model."""

    from peft import PeftModel

    path = Path(adapter_directory)
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Antidoom adapter config not found: {path / 'adapter_config.json'}")
    adapter_model = PeftModel.from_pretrained(base_model, path, is_trainable=is_trainable)
    if is_trainable:
        _validate_antidoom_trainable_parameters(adapter_model)
    return adapter_model


def apply_configured_antidoom_rwkv_adapter(model: torch.nn.Module, model_config: Any) -> torch.nn.Module:
    """Apply the explicit native-model adapter config during model construction."""

    value = configured_antidoom_rwkv_adapter(model_config)
    if value is None:
        return model

    adapter_path = value.get("path") or value.get("adapter_path")
    merge = bool(value.get("merge", False))
    if merge and not adapter_path:
        raise ValueError("Antidoom RWKV adapter merge requires lora.path")
    if adapter_path:
        adapter_model = load_antidoom_rwkv_adapter(model, adapter_path, is_trainable=not merge)
    else:
        adapter_model = inject_antidoom_rwkv_adapter(model, AntidoomRWKVAdapterConfig.from_mapping(value))
    return adapter_model.merge_and_unload(safe_merge=True) if merge else adapter_model


def antidoom_trainable_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """Return the adapter-only optimizer parameter selection."""

    selected = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not selected:
        raise RuntimeError("Antidoom RWKV adapter has no trainable parameters")
    return selected


def restrict_optimizer_to_antidoom_adapter(optimizer: Any, model: torch.nn.Module) -> int:
    """Remove frozen base tensors from a native rwkv-lm optimizer in place."""

    selected = antidoom_trainable_parameters(model)
    selected_ids = {id(parameter) for _, parameter in selected}
    retained_ids: set[int] = set()
    for group in optimizer.param_groups:
        group["params"] = [parameter for parameter in group["params"] if id(parameter) in selected_ids]
        retained_ids.update(id(parameter) for parameter in group["params"])
    for parameter in list(optimizer.state):
        if id(parameter) not in selected_ids:
            del optimizer.state[parameter]
    missing = selected_ids - retained_ids
    if missing:
        raise RuntimeError(f"native RWKV optimizer omitted {len(missing)} Antidoom adapter parameters")
    return sum(parameter.numel() for _, parameter in selected)


def save_antidoom_rwkv_adapter(model: torch.nn.Module, output_directory: str | Path) -> Path:
    """Save only LoRA tensors and PEFT metadata in the community format."""

    from peft import PeftModel

    if not isinstance(model, PeftModel):
        raise TypeError("save_antidoom_rwkv_adapter requires a PEFT model")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    required = (output / "adapter_config.json", output / "adapter_model.safetensors")
    if any(not path.is_file() for path in required):
        raise RuntimeError(f"incomplete Antidoom adapter save under {output}")
    return output


def merge_antidoom_rwkv_adapter(
    base_model: torch.nn.Module,
    adapter_directory: str | Path,
) -> torch.nn.Module:
    """Load an adapter into a clean base model and permanently merge its delta."""

    adapter_model = load_antidoom_rwkv_adapter(base_model, adapter_directory, is_trainable=False)
    merged_model = adapter_model.merge_and_unload(safe_merge=True)
    if any("lora_" in name for name in merged_model.state_dict()):
        raise RuntimeError("merged Antidoom RWKV model still contains LoRA tensors")
    return merged_model


def save_merged_rwkv_checkpoint(model: torch.nn.Module, output_path: str | Path) -> Path:
    """Atomically save a native ``.pth`` checkpoint reloadable without PEFT."""

    state = {}
    for name, value in model.state_dict().items():
        normalized_name = name.replace("base_model.model.", "").replace(".base_layer", "")
        if "lora_" in normalized_name:
            raise ValueError("merge the Antidoom adapter before saving a native RWKV checkpoint")
        state[normalized_name] = value.detach().cpu()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.save(state, temporary)
    temporary.replace(output)
    return output


def antidoom_ftpo_loss(
    logits: torch.Tensor,
    *,
    chosen_token_ids: torch.Tensor,
    rejected_token_ids: torch.Tensor,
    chosen_mask: torch.Tensor | None = None,
    reference_logits: torch.Tensor | None = None,
    clip_epsilon: float = 2.0,
    lambda_mse: float = 0.4,
    lambda_mse_target: float = 0.05,
    target_tolerance: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the final-token preference objective and auditable eval metrics."""

    if logits.ndim != 2:
        raise ValueError("Antidoom FTPO logits must have shape [batch, vocabulary]")
    if chosen_token_ids.ndim == 1:
        chosen_token_ids = chosen_token_ids.unsqueeze(-1)
    if chosen_mask is None:
        chosen_mask = torch.ones_like(chosen_token_ids, dtype=torch.bool)
    else:
        chosen_mask = chosen_mask.to(dtype=torch.bool)
    if chosen_token_ids.shape != chosen_mask.shape or rejected_token_ids.shape != logits.shape[:1]:
        raise ValueError("Antidoom FTPO token ids and masks do not match the logits batch")
    if not torch.any(chosen_mask):
        raise ValueError("Antidoom FTPO requires at least one active chosen token")
    if clip_epsilon <= 0:
        raise ValueError("Antidoom FTPO clip_epsilon must be positive")

    row_ids = torch.arange(logits.size(0), device=logits.device).unsqueeze(1)
    rejected_logits = logits.gather(-1, rejected_token_ids.unsqueeze(-1))
    deltas = logits[row_ids, chosen_token_ids] - rejected_logits
    active_weights = torch.clamp((clip_epsilon - deltas) / clip_epsilon, min=0.0, max=1.0) * chosen_mask
    chosen_counts = chosen_mask.sum(dim=-1).clamp(min=1)
    preference_loss = ((F.softplus(clip_epsilon - deltas) * active_weights).sum(dim=-1) / chosen_counts).mean()
    loss = preference_loss

    mse_other = logits.new_tensor(0.0)
    mse_target = logits.new_tensor(0.0)
    if reference_logits is not None:
        if reference_logits.shape != logits.shape:
            raise ValueError("Antidoom FTPO reference_logits must match logits")
        difference = logits - reference_logits.detach()
        target_mask = torch.zeros_like(logits, dtype=torch.bool)
        target_mask[row_ids.expand_as(chosen_token_ids)[chosen_mask], chosen_token_ids[chosen_mask]] = True
        target_mask.scatter_(1, rejected_token_ids.unsqueeze(-1), True)
        other_mask = ~target_mask
        mse_other = (difference.square() * other_mask).sum() / other_mask.sum().clamp(min=1)
        excess = torch.clamp(difference.abs() - target_tolerance, min=0.0)
        mse_target = (excess.square() * target_mask).sum() / target_mask.sum().clamp(min=1)
        loss = loss + lambda_mse * mse_other + lambda_mse_target * mse_target

    active_deltas = deltas[chosen_mask]
    log_probabilities = F.log_softmax(logits, dim=-1)
    chosen_log_probabilities = log_probabilities[row_ids, chosen_token_ids]
    rejected_log_probabilities = log_probabilities.gather(-1, rejected_token_ids.unsqueeze(-1))
    wins = (chosen_log_probabilities > rejected_log_probabilities) & chosen_mask
    metrics = {
        "pref_loss": preference_loss.detach(),
        "chosen_win": (wins.sum(dim=-1) / chosen_counts).float().mean().detach(),
        "margin_win": ((deltas >= clip_epsilon) & chosen_mask).sum().float().div(chosen_mask.sum()).detach(),
        "mean_delta": active_deltas.mean().detach(),
        "median_delta": active_deltas.median().detach(),
        "active_weight": active_weights[chosen_mask].mean().detach(),
        "mse_elem": mse_other.detach(),
        "mse_tgt_tokenwise": mse_target.detach(),
    }
    return loss, metrics


def evaluate_antidoom_rwkv(
    model: torch.nn.Module,
    prompt_token_ids: torch.Tensor,
    *,
    chosen_token_ids: torch.Tensor,
    rejected_token_ids: torch.Tensor,
    chosen_mask: torch.Tensor | None = None,
    clip_epsilon: float = 2.0,
) -> dict[str, float]:
    """Run the native RWKV tensor forward and report FTPO preference metrics."""

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(prompt_token_ids)
            logits = _extract_logits(output)
            _, metrics = antidoom_ftpo_loss(
                logits[:, -1, :],
                chosen_token_ids=chosen_token_ids,
                rejected_token_ids=rejected_token_ids,
                chosen_mask=chosen_mask,
                clip_epsilon=clip_epsilon,
                lambda_mse=0.0,
                lambda_mse_target=0.0,
            )
    finally:
        model.train(was_training)
    return {name: float(value.item()) for name, value in metrics.items()}


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping) and isinstance(output.get("logits"), torch.Tensor):
        return output["logits"]
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(output, Sequence) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Antidoom RWKV eval requires model output logits")


def _validate_antidoom_trainable_parameters(model: torch.nn.Module) -> None:
    selected = antidoom_trainable_parameters(model)
    offenders = [name for name, _ in selected if "lora_" not in name]
    if offenders:
        raise RuntimeError("Antidoom RWKV base parameters remained trainable: " + ",".join(offenders[:20]))
